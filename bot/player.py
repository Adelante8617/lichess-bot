"""对局阶段的决策：复杂度分流 → LLM 选着（可调工具）→ 落子前自检 → 非法着法重试 / 保底。"""
import json
import random

import chess

from . import board_view
from .boardtext import (COLOR_ZH, board_meta, describe_last_move, display_san, legal_san_map,
                        material_lead, parse_model_move, piece_lists, render_board, san_history,
                        strip_check)
from .config import (ANALYSIS_BOARD, BOARD_RELATIONS, COMPLEXITY_CHECK, COMPLEXITY_DEFAULT,
                     COMPLEXITY_PROFILE, LLM_MAX_TOKENS, MATERIAL_LEAD_EFFORT, MATERIAL_LEAD_SKIP,
                     OPENING_EFFORT, OPENING_FAST_MOVES, PLAN_MEMORY, SELF_CHECK_EFFORT,
                     SELF_CHECK_ROUNDS, TOOL_ROUNDS)
from .live import live
from .llm import cap_ladder, extract_json, llm_call, reasoning_of, think_ladder
from .memory import recall_experience
from .prompts import self_check_prompt, system_prompt
from .tools import play_tools, run_tool


def fallback_move(legal_moves: list[str]) -> str:
    """模型多次给出非法走法时的最后保底：随机选一个合法着法，
    不做任何局面判断（杀棋 / 吃子 / 安全性都不替模型看），只保证不因超时或非法着法判负。"""
    return random.choice(legal_moves)


# 模型自己的长期计划：{我方颜色: (定下计划时的着法序列 UCI, 计划)}。
# 新局面若不是在该序列基础上继续（换了一盘棋），计划自动作废。
_plans: dict[bool, tuple[list[str], str]] = {}


def recall_plan(board: chess.Board) -> str:
    stored = _plans.get(board.turn)
    if not stored:
        return ""
    moves, plan = stored
    now = [m.uci() for m in board.move_stack]
    return plan if now[:len(moves)] == moves else ""


OBS_KEYS = ["complexity", "complexity_reason", "my_attacked", "opp_attacked", "my_hanging",
            "opp_hanging", "check_chance", "capture_chance", "threats", "tactics",
            "candidates", "pv", "board_summary"]


def self_check(board: chess.Board, messages: list, move: chess.Move,
               legal_sans: list[str], complexity: str, levels: list[str] | None = None,
               max_tokens: int | None = None) -> tuple[chess.Move, list[dict]]:
    """落子前自检：让模型站在对方角度重新审视选定着法，发现会白丢子就换。
    只让模型自己复查，程序不做任何局面判断。返回 (最终着法, 每轮自检记录)。

    前几轮否决过的着法（连同否决理由）会带进后续轮次，且不允许改回去：
    否则第二轮看不到第一轮的结论，会出现 Qxh2 → Qxd4 → Qxh2 这样的来回摇摆。"""
    records = []
    current = move
    rejected: dict[chess.Move, str] = {}  # 已否决的着法 → 否决理由
    for rnd in range(1, SELF_CHECK_ROUNDS + 1):
        san = display_san(board, current)
        after = board.copy()
        after.push(current)
        live.stage(f"第 {rnd} 轮自检：复查 {san}")
        relations = board_view.relations_text(after) if BOARD_RELATIONS else ""
        rejected_view = {display_san(board, m): r for m, r in rejected.items()}
        prompt = self_check_prompt(san, render_board(after), COLOR_ZH[after.turn], relations,
                                   complexity, legal_sans, rejected_view)
        msgs = messages + [{"role": "user", "content": prompt}]
        try:
            msg, _ = llm_call(msgs, levels=levels, max_tokens=max_tokens)
        except Exception as e:
            print(f"[SELF-CHECK] LLM failed: {e}")
            break
        content = (msg.content or "").strip()
        print(f"[SELF-CHECK round={rnd}] {content}")
        obj = extract_json(content) or {}
        rec = {"round": rnd, "checked": san,
               **{k: str(obj.get(k, "")) for k in
                  ("opp_best_reply", "danger", "material_after", "verdict", "move", "reason")},
               "reasoning": reasoning_of(msg)}
        records.append(rec)
        if str(obj.get("verdict", "")).strip().lower() != "change":
            break
        new = parse_model_move(board, str(obj.get("move", "")))
        if new is None or new == current:
            rec["note"] = "改选的着法无效或与原着法相同，保持原着法"
            print(f"[SELF-CHECK] {rec['note']}: {obj.get('move')!r}")
            break
        if new in rejected:
            # 想改回之前否决过的着法：两步都被判定有问题，保留本轮刚复查过的这步，不再摇摆
            rec["note"] = (f"想改回已否决的 {display_san(board, new)}（理由：{rejected[new]}），"
                           f"拒绝改回，保持 {san}")
            print(f"[SELF-CHECK] {rec['note']}")
            break
        rejected[current] = rec["reason"] or rec["danger"] or "自检判定有问题"
        print(f"[SELF-CHECK] 改选 {san} -> {display_san(board, new)}")
        rec["changed_to"] = display_san(board, new)
        current = new
        if rnd == SELF_CHECK_ROUNDS:
            # 自检轮数用完，新着法没有再被复查；它仍好过已被否决的着法，照常采用，但标明出来
            rec["note"] = f"自检轮数已用完，{rec['changed_to']} 未经复查"
            print(f"[SELF-CHECK] {rec['note']}")
    return current, records


def is_opening_fast(board: chess.Board, prev_board: chess.Board | None,
                    opp_last_move: str | None) -> bool:
    """开局快速模式：前 OPENING_FAST_MOVES 个全回合内，且未被将军、对方上一步不是吃子。"""
    if board.fullmove_number > OPENING_FAST_MOVES or board.is_check():
        return False
    if prev_board is not None and opp_last_move:
        try:
            if prev_board.is_capture(chess.Move.from_uci(opp_last_move)):
                return False
        except ValueError:
            pass
    return True


def classify_complexity(board: chess.Board, last_section: str, legal_sans: list[str]) -> tuple[str, str]:
    """独立的一次不思考调用，只判断当前局面复杂度，返回 (simple/medium/complex, 理由)。
    程序只提供盘面与双方着法列表等原始事实，判断由模型做；失败时返回 COMPLEXITY_DEFAULT。"""
    opp_sans = "（我方正被将军，略）"
    if not board.is_check():
        tmp = board.copy()
        tmp.push(chess.Move.null())
        opp_sans = ", ".join(legal_san_map(tmp))
    recent = " ".join(san_history(board).split()[-12:]) or "（尚无着法）"
    prompt = f"""只判断当前局面的复杂度，不要选着、不要计算变化。
盘面(白=W*, 黑=B*, '.'=空，第二个字母为子种 K/Q/R/B/N/P)，轮到{COLOR_ZH[board.turn]}走：
{render_board(board)}

对方刚走的一步：{last_section}
最近着法：{recent}
{board_meta(board)}

我方合法走法（带 + 为将军）：{", ".join(legal_sans)}
假如轮到对方走，对方的走法：{opp_sans}

分级标准：
- simple：常规出子/调动，双方都没有吃子或将军的着法；或只有一个明显应着（必须应将、必须吃回被兑的子）。
- medium：常规中局，有若干合理计划，双方子力有接触但没有直接战术。
- complex：存在吃子、将军、捉双、牵制、悬子、王翼攻击等直接战术，或残局需要精确计算。
只要双方任一方有吃子或将军的着法，就不是 simple。

严格输出 JSON（不要 markdown）：{{"complexity": "simple/medium/complex", "reason": "≤30 字"}}"""
    live.stage("判断局面复杂度")
    try:
        msg, _ = llm_call([{"role": "user", "content": prompt}], levels=["off"], max_tokens=512)
        obj = extract_json((msg.content or "").strip()) or {}
    except Exception as e:
        print(f"[COMPLEXITY] LLM failed: {e}")
        obj = {}
    level = str(obj.get("complexity", "")).strip().lower()
    reason = str(obj.get("reason", "")).strip()
    if level not in COMPLEXITY_PROFILE:
        print(f"[COMPLEXITY] 无法解析 {obj!r}，使用默认 {COMPLEXITY_DEFAULT}")
        return COMPLEXITY_DEFAULT, "复杂度判断失败，使用默认档"
    return level, reason


def get_llm_move(board: chess.Board, ply: int, prev_board: chess.Board | None,
                 opp_last_move: str | None,
                 chat_messages: list | None = None):
    # 注意：教练在聊天框中的实时评价【只在赛后复盘】使用，
    # 对局进行中模型看不到，避免实时作弊式指导。chat_messages 仅作签名兼容。
    _ = chat_messages
    san_map = legal_san_map(board)
    if not san_map:
        return None, "no legal move", "", {}
    legal_sans = list(san_map)

    # 我方 / 对方 颜色字符串
    my_color_str = COLOR_ZH[board.turn] + ("(WHITE)" if board.turn == chess.WHITE else "(BLACK)")
    white_pieces, black_pieces = piece_lists(board)
    my_pieces = white_pieces if board.turn == chess.WHITE else black_pieces
    opp_pieces = black_pieces if board.turn == chess.WHITE else white_pieces

    if prev_board is not None and opp_last_move:
        last_section = describe_last_move(prev_board, opp_last_move)
    else:
        last_section = "（开局第一手，对方尚未走子）"

    live.thinking(ply)
    fast = is_opening_fast(board, prev_board, opp_last_move)
    # 本步复杂度 → 起始思考档位与 max_tokens
    complexity, complexity_reason = "", ""
    if fast:
        print(f"[FAST] 开局快速模式（fullmove={board.fullmove_number} ≤ {OPENING_FAST_MOVES}）")
        complexity, complexity_reason = "simple", "开局快速模式"
        ladder = think_ladder(OPENING_EFFORT)
        max_tokens = COMPLEXITY_PROFILE.get("simple", [None, LLM_MAX_TOKENS])[1]
    elif MATERIAL_LEAD_SKIP > 0 and (lead := material_lead(board)) >= MATERIAL_LEAD_SKIP:
        # 子力大幅领先：不调 LLM 判断复杂度，直接用较低档位；复杂度标签取起始档位相同的那一档
        complexity = next((k for k, (e, _) in COMPLEXITY_PROFILE.items() if e == MATERIAL_LEAD_EFFORT),
                          COMPLEXITY_DEFAULT)
        complexity_reason = f"我方子力领先 {lead} 分，跳过复杂度判断"
        ladder = think_ladder(MATERIAL_LEAD_EFFORT)
        max_tokens = COMPLEXITY_PROFILE.get(complexity, [None, LLM_MAX_TOKENS])[1]
        print(f"[COMPLEXITY] {complexity_reason} -> effort={MATERIAL_LEAD_EFFORT}, max_tokens={max_tokens}")
    elif COMPLEXITY_CHECK:
        complexity, complexity_reason = classify_complexity(board, last_section, legal_sans)
        start, max_tokens = COMPLEXITY_PROFILE[complexity]
        ladder = think_ladder(start)
        print(f"[COMPLEXITY] {complexity}（{complexity_reason}）-> effort={start}, max_tokens={max_tokens}")
    else:
        ladder, max_tokens = think_ladder(), LLM_MAX_TOKENS
    recalled = [] if fast else recall_experience(board)
    if recalled:
        recall_section = "\n".join(f"- {h['text']}" for h in recalled)
        print(f"[RECALL] {len(recalled)} 条: " + " | ".join(h["text"][:40] for h in recalled))
    else:
        recall_section = "（无）"

    full_move = (ply + 1) // 2  # 1-based 回合
    history = san_history(board) or "（尚无着法）"

    aid_section = ""
    if BOARD_RELATIONS:
        aid_section += ("\n==== 子力关系（程序按规则列出的原始事实，不含任何判断）====\n"
                        f"{board_view.relations_text(board)}\n")
    if PLAN_MEMORY:
        prev_plan = recall_plan(board)
        aid_section += ("\n==== 你之前定下的计划 ====\n"
                        f"{prev_plan or '（尚无，请根据局面制定）'}\n")

    user_prompt = f"""【当前局面，轮到你走】
盘面(白=W*, 黑=B*, '.'=空，第二个字母为子种 K/Q/R/B/N/P):
{render_board(board)}

【对方刚走的一步】
{last_section}

【对局至今的全部着法（SAN）】
{history}
（可据此回顾双方计划、你自己此前的布局意图，并留意是否在重复局面）

==== 身份与子力 ====
你执 {my_color_str}，只能移动自己的子。
我方子力: {my_pieces}
对方子力: {opp_pieces}

==== 局面信息 ====
{board_meta(board)}
回合(ply): {ply}  全回合数(fullmove): {full_move}
{aid_section}
==== 经验库自动召回（仅供参考，与当前局面不符就忽略）====
{recall_section}

合法走法（SAN，已替你过滤，只含你能走的着；带 + 表示该着会将军）:
{", ".join(legal_sans)}

要求：
{"- 【开局快速模式】这是常规开局阶段：按开局原则（或调用 search_opening_book）快速选着，不要长时间计算；complexity 填 simple，think ≤60 字，pv 给 3 步即可；只需确认所走的子不会被白吃。" if fast else
  f"- 本步局面复杂度已单独判定为 {complexity}（{complexity_reason}），按此档决定思考投入（见系统提示第 0 节），complexity 字段照填 {complexity}；" if complexity else
  "- 先判断局面复杂度，按复杂度决定思考投入（见系统提示第 0 节）；"}
- 落子前完成系统提示 D 节的安全检查；
- move 必须逐字取自上面的合法走法列表，且等于 pv[0]；candidates / pv 全部用 SAN。

请按系统提示输出完整 JSON。"""

    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt},
    ]

    think, opp_intent, reasoning = "", "", ""
    obs: dict = {}
    warnings: list[str] = []
    recalled_view = [{"text": h["text"], "score": round(h["score"], 3)} for h in recalled]

    MAX_ATTEMPTS = 3
    step = 0  # 当前档位在 ladder 中的下标；被截断或走法非法都会往下降
    for attempt in range(1, MAX_ATTEMPTS + 1):
        content = ""
        live.stage(f"第 {attempt} 次决策" if attempt > 1 else "思考中")
        for rnd in range(TOOL_ROUNDS):  # 单轮内最多 TOOL_ROUNDS 次 tool 往返
            # 开分析棋盘时，最后一轮不再提供工具，逼模型给出最终答案
            last_round = ANALYSIS_BOARD and rnd == TOOL_ROUNDS - 1
            msg, level = llm_call(messages, tools=None if last_round else play_tools(),
                                  levels=ladder[step:], max_tokens=max_tokens)
            step = ladder.index(level)  # 若本轮因截断降了档，后续沿用降后的档位
            if msg.tool_calls:
                assistant = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                }
                if reasoning_of(msg):  # 思考模型在工具往返中需要带回推理过程
                    assistant["reasoning_content"] = reasoning_of(msg)
                messages.append(assistant)
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    result = run_tool(tc.function.name, args, {"board": board})
                    print(f"[TOOL] {tc.function.name}({args}) -> {result[:120]}")
                    live.tool(tc.function.name, args, result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue
            content = (msg.content or "").strip()
            reasoning = reasoning_of(msg) or reasoning
            # 把这条 assistant 消息加入历史，便于后续反馈 / 自检
            messages.append({"role": "assistant", "content": content})
            break

        if reasoning:
            print(f"[REASONING attempt={attempt}] {len(reasoning)} 字")
        print(f"[LLM raw attempt={attempt}] {content}")

        cur_think, cur_move, cur_opp = "", "", ""
        cur_obs: dict = {}
        obj = extract_json(content)
        if obj:
            cur_opp = str(obj.get("opp_intent", "")).strip()
            cur_think = str(obj.get("think", "")).strip()
            cur_move = str(obj.get("move", "")).strip()
            cur_obs = {k: obj.get(k, "") for k in OBS_KEYS}
            if complexity:  # 以独立判断的复杂度为准
                cur_obs["complexity"], cur_obs["complexity_reason"] = complexity, complexity_reason
            if PLAN_MEMORY:
                cur_obs["plan"] = str(obj.get("plan", "")).strip()
            pv = cur_obs.get("pv")
            if cur_move and isinstance(pv, list) and pv \
                    and strip_check(str(pv[0])) != strip_check(cur_move):
                print(f"[WARN] move={cur_move} 与 pv[0]={pv[0]} 不一致，以 move 为准")
                warnings.append(f"move={cur_move} 与 pv[0]={pv[0]} 不一致")

        # 只要当前轮抓到了任何字段，就更新（即便最终走法非法，思考过程也保留最新）
        if cur_think:
            think = cur_think
        if cur_opp:
            opp_intent = cur_opp
        if cur_obs:
            obs = cur_obs

        mv = parse_model_move(board, cur_move)
        if mv is not None:
            chosen_san = display_san(board, mv)
            checks: list[dict] = []
            if SELF_CHECK_ROUNDS > 0 and not fast:
                mv, checks = self_check(board, messages, mv, legal_sans, str(obs.get("complexity", "")),
                                        levels=cap_ladder(ladder[step:], SELF_CHECK_EFFORT),
                                        max_tokens=max_tokens)
                obs["self_check"] = checks
                if display_san(board, mv) != chosen_san:
                    warnings.append(f"自检后改选：{chosen_san} → {display_san(board, mv)}")
                    obs["pv"] = []  # 原主变基于旧着法，已失效
            if PLAN_MEMORY and obs.get("plan"):
                _plans[board.turn] = ([m.uci() for m in board.move_stack] + [mv.uci()], obs["plan"])
            live.decision(ply, move_san=display_san(board, mv), move_uci=mv.uci(),
                          first_choice=chosen_san, think=think, opp_intent=opp_intent, obs=obs,
                          reasoning=reasoning, recalled=recalled_view,
                          warnings=warnings + (["开局快速模式：已跳过自检"] if fast else []),
                          attempts=attempt, fallback=False)
            return mv.uci(), think, opp_intent, obs

        # 非法 / 缺失 → 给模型反馈，要求重想
        reason = "未给出 move 字段" if not cur_move else f"'{cur_move}' 不是当前局面的合法着法"
        print(f"[WARN] attempt {attempt}: illegal move ({reason})")
        warnings.append(f"第 {attempt} 次输出无效：{reason}")
        if attempt < MAX_ATTEMPTS:
            step = min(step + 1, len(ladder) - 1)
            # 只保留 system/user 初始提示 + 上一次回答，丢弃 tool 往返，防止重试时上下文膨胀
            messages = messages[:2] + [{"role": "assistant", "content": content}]
            messages.append({
                "role": "user",
                "content": (
                    f"你刚才输出的走法非法：{reason}。\n"
                    f"请重新思考。注意：必须从下面的合法走法列表中选一个（SAN 记谱，"
                    f"升变写法如 e8=Q）：\n"
                    f"{', '.join(legal_sans)}\n"
                    f"再次按系统提示输出完整 JSON。"
                ),
            })
            continue

    # 三次都失败：fallback
    move = fallback_move([m.uci() for m in san_map.values()])
    print(f"[WARN] all {MAX_ATTEMPTS} attempts illegal, random legal fallback -> {move}")
    live.decision(ply, move_san=display_san(board, chess.Move.from_uci(move)), move_uci=move,
                  think=think, opp_intent=opp_intent, obs=obs, reasoning=reasoning,
                  recalled=recalled_view,
                  warnings=warnings + ["三次均无效，随机选择合法着法保底"],
                  attempts=MAX_ATTEMPTS, fallback=True)
    return move, think, opp_intent, obs
