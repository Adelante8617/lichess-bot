"""对局阶段的决策：复杂度分流 → LLM 选着（可调工具）→ 落子前自检 → 非法着法重试 / 保底。"""
import json
from concurrent.futures import ThreadPoolExecutor
import random

import chess

from . import board_view
from .boardtext import (COLOR_ZH, board_meta, describe_last_move, display_san, legal_san_map,
                        material_lead, parse_model_move, piece_lists, render_board, san_history,
                        strip_check)
from .config import (ANALYSIS_BOARD, BOARD_RELATIONS, COMPLEXITY_CHECK, COMPLEXITY_DEFAULT,
                     COMPLEXITY_PROFILE, HANG_GUARD, HANG_GUARD_MIN, HANG_GUARD_POSITIONAL,
                     HANG_GUARD_ROUNDS, LLM_MAX_TOKENS, MATERIAL_LEAD_EFFORT,
                     MATERIAL_LEAD_SKIP, OPENING_EFFORT, OPENING_FAST_MOVES, PLAN_MEMORY,
                     SELF_CHECK_EFFORT, SELF_CHECK_ROUNDS, STRATEGY_STAGE, STRATEGY_STAGE_MAX_TOKENS,
                     TOOL_ROUNDS)
from .archive import log_reasoning
from .guard import material_risk, risk_text, verify_line
from .live import live
from .llm import cap_ladder, extract_json, llm_call, reasoning_of, think_ladder
from .memory import recall_experience
from .prompts import (STRATEGY_STAGE_PROMPT, complexity_prompt, hang_guard_prompt, self_check_prompt,
                      strategy_stage_section, system_prompt, user_prompt)
from .tools import play_tools, run_tool


def fallback_move(legal_moves: list[str]) -> str:
    """模型多次给出非法走法时的最后保底：随机选一个合法着法，
    不做任何局面判断（杀棋 / 吃子 / 安全性都不替模型看），只保证不因超时或非法着法判负。"""
    return random.choice(legal_moves)


# 模型自己定下的战略方针：{我方颜色: (定下方针时的着法序列 UCI, 方针)}。
# 新局面若不是在该序列基础上继续（换了一盘棋），方针自动作废。
_plans: dict[bool, tuple[list[str], str]] = {}


def recall_plan(board: chess.Board) -> str:
    stored = _plans.get(board.turn)
    if not stored:
        return ""
    moves, plan = stored
    now = [m.uci() for m in board.move_stack]
    return plan if now[:len(moves)] == moves else ""


OBS_KEYS = ["complexity", "complexity_reason", "urgent", "strategy",
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
        log_reasoning(f"SELF-CHECK round={rnd}", reasoning_of(msg))
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


def hang_guard(board: chess.Board, messages: list, move: chess.Move, legal_sans: list[str],
               candidates: list[str], levels: list[str] | None = None,
               max_tokens: int | None = None) -> tuple[chess.Move, list[dict]]:
    """丢子守卫：程序模拟对方吃子交换，单格交换会净亏 ≥ HANG_GUARD_MIN 分时把模拟结果交给模型复查。
    - 模型坚持并给出变化（tactical）：程序按规则摆一遍，合法且终点拿回子力 / 将杀就照走；
      摆不通就把摆出的事实交回模型，进入下一轮。
    - 模型坚持并说明局面性补偿（positional）：净亏 ≤ HANG_GUARD_POSITIONAL 时照走，不要求变化。
    - 模型改选：新着法同样要过守卫。
    轮数用完仍未解决时，从模型自己的 candidates 里挑第一个不丢子的着法（都丢子则保持原着法）。
    每条记录带 outcome（kept_tactical / kept_positional / changed / fallback / unresolved），供赛后统计。"""
    records: list[dict] = []
    current = move
    rejected: dict[chess.Move, str] = {}  # 模型自己放弃的、会丢子的着法 → 模拟结果
    feedback = ""
    for rnd in range(1, HANG_GUARD_ROUNDS + 1):
        risk = material_risk(board, current)
        if risk["loss"] < HANG_GUARD_MIN:
            return current, records
        san = display_san(board, current)
        fact = risk_text(board, current, risk)
        positional_ok = risk["loss"] <= HANG_GUARD_POSITIONAL
        print(f"[GUARD round={rnd}] {fact}")
        live.stage(f"丢子守卫：复查 {san}")
        rejected_view = {display_san(board, m): r for m, r in rejected.items()}
        prompt = hang_guard_prompt(san, fact, legal_sans, rejected_view, positional_ok, feedback)
        try:
            msg, _ = llm_call(messages + [{"role": "user", "content": prompt}], levels=levels,
                              max_tokens=max_tokens)
        except Exception as e:
            print(f"[GUARD] LLM failed: {e}")
            break
        content = (msg.content or "").strip()
        print(f"[GUARD reply round={rnd}] {content}")
        log_reasoning(f"GUARD round={rnd}", reasoning_of(msg))
        obj = extract_json(content) or {}
        rec = {"round": rnd, "checked": san, "checked_uci": current.uci(), "fact": fact,
               "loss": risk["loss"],
               **{k: str(obj.get(k, "")).strip() for k in ("verdict", "kind", "move", "line", "reason")},
               "reasoning": reasoning_of(msg)}
        records.append(rec)
        feedback = ""
        if rec["verdict"].lower() == "keep":
            if rec["line"]:
                check = verify_line(board, current, rec["line"], HANG_GUARD_MIN, risk["square"])
                rec["line_check"] = check["text"]
                print(f"[GUARD] 核对变化：{check['text']}")
                if check["ok"]:
                    rec["outcome"], rec["note"] = "kept_tactical", f"保持 {san}，变化核对通过"
                    return current, records
                feedback = f"{rec['line']} → {check['text']}"
            if positional_ok and rec["kind"].lower() == "positional" and rec["reason"]:
                rec["outcome"], rec["note"] = "kept_positional", f"保持 {san}，局面性弃子：{rec['reason']}"
                print(f"[GUARD] {rec['note']}")
                return current, records
            if not feedback:
                feedback = "上一轮选择保持，但没有给出变化。"
            continue  # 同一着法再给一轮，带上摆出的事实
        new = parse_model_move(board, rec["move"])
        if new is None or new == current or new in rejected:
            rec["note"] = f"改选的着法无效或已放弃过：{rec['move']!r}"
            break
        rejected[current] = fact
        rec["outcome"], rec["changed_to"] = "changed", display_san(board, new)
        print(f"[GUARD] 改选 {san} -> {rec['changed_to']}")
        current = new
    if material_risk(board, current)["loss"] < HANG_GUARD_MIN:
        return current, records
    rejected[current] = ""
    for text in candidates:
        mv = parse_model_move(board, text)
        if mv is not None and mv not in rejected and material_risk(board, mv)["loss"] < HANG_GUARD_MIN:
            outcome, note = "fallback", f"复查后仍未解决，改用候选里不丢子的 {display_san(board, mv)}"
            break
    else:
        mv = current
        outcome, note = "unresolved", "复查后仍未解决，但候选里没有不丢子的着法，保持原着法"
    print(f"[GUARD] {note}")
    records.append({"round": len(records) + 1, "checked": display_san(board, current),
                    "checked_uci": current.uci(), "outcome": outcome, "note": note,
                    **({"changed_to": display_san(board, mv)} if mv != current else {})})
    return mv, records


def strategy_stage(board: chess.Board, messages: list) -> dict | None:
    """第一阶段：关闭思考，只定紧急情况 / 战略方针 / ≤3 个候选。
    不给思考空间，模型就没法把合法着法逐个试一遍。失败或没有合法候选时返回 None（退回单阶段）。"""
    live.stage("第一阶段：定方针与候选")
    try:
        msg, _ = llm_call(messages + [{"role": "user", "content": STRATEGY_STAGE_PROMPT}],
                          levels=["off"], max_tokens=STRATEGY_STAGE_MAX_TOKENS)
    except Exception as e:
        print(f"[STAGE1] LLM failed: {e}")
        return None
    obj = extract_json((msg.content or "").strip()) or {}
    candidates = []
    for c in obj.get("candidates") or []:
        mv = parse_model_move(board, str(c.get("move", ""))) if isinstance(c, dict) else None
        if mv is not None and all(x["move"] != display_san(board, mv) for x in candidates):
            candidates.append({"move": display_san(board, mv), "purpose": str(c.get("purpose", "")),
                               "idea": str(c.get("idea", ""))})
    if not candidates:
        print(f"[STAGE1] 没有合法候选，退回单阶段: {obj!r}")
        return None
    stage = {"urgent": str(obj.get("urgent", "")).strip(), "strategy": str(obj.get("strategy", "")).strip(),
             "candidates": candidates[:3]}
    print(f"[STAGE1] 方针: {stage['strategy']} | 候选: {', '.join(c['move'] for c in stage['candidates'])}")
    return stage


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
    prompt = complexity_prompt(side=COLOR_ZH[board.turn], board_text=render_board(board),
                               last_move=last_section, recent=recent, meta=board_meta(board),
                               legal_sans=legal_sans, opp_sans=opp_sans)
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
    # 经验召回要做一次 embedding，与复杂度判断的 LLM 调用互不依赖，并行跑
    recall_pool = ThreadPoolExecutor(max_workers=1)
    recall_future = None if fast else recall_pool.submit(recall_experience, board)
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
    recalled = recall_future.result() if recall_future else []
    recall_pool.shutdown(wait=False)
    if recalled:
        recall_section = "\n".join(f"- {h['text']}" for h in recalled)
        print(f"[RECALL] {len(recalled)} 条: " + " | ".join(h["text"][:40] for h in recalled))
    else:
        recall_section = "（无）"

    prompt = user_prompt(
        board_text=render_board(board), last_move=last_section,
        history=san_history(board) or "（尚无着法）", my_color=my_color_str,
        my_pieces=my_pieces, opp_pieces=opp_pieces, meta=board_meta(board), ply=ply,
        relations=board_view.relations_text(board) if BOARD_RELATIONS else "",
        prev_strategy=recall_plan(board) if PLAN_MEMORY else None,
        recalled=recall_section, legal_sans=legal_sans, fast=fast,
        complexity=complexity, complexity_reason=complexity_reason)

    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": prompt},
    ]
    stage = strategy_stage(board, messages) if STRATEGY_STAGE and not fast else None
    if stage:
        # 附在 user 提示末尾（而不是新增消息），非法着法重试时保留的 messages[:2] 里也有它
        messages[1]["content"] += strategy_stage_section(stage["urgent"], stage["strategy"],
                                                         stage["candidates"])

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
            msg, level = llm_call(messages, tools=None if last_round or fast else play_tools(),
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
            log_reasoning(f"attempt={attempt}", reasoning)
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
            cur_obs["strategy"] = str(cur_obs.get("strategy") or (stage or {}).get("strategy", "")).strip()
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
            check_levels = cap_ladder(ladder[step:], SELF_CHECK_EFFORT)
            guard_cands = [str(c.get("move", "")) for c in list(obs.get("candidates") or [])
                           + (stage or {}).get("candidates", []) if isinstance(c, dict)]
            guards: list[dict] = []
            if HANG_GUARD:
                mv, guards = hang_guard(board, messages, mv, legal_sans, guard_cands,
                                        levels=check_levels, max_tokens=max_tokens)
            if SELF_CHECK_ROUNDS > 0 and not fast:
                before = mv
                mv, checks = self_check(board, messages, mv, legal_sans, str(obs.get("complexity", "")),
                                        levels=check_levels, max_tokens=max_tokens)
                obs["self_check"] = checks
                if HANG_GUARD and mv != before:  # 自检改出来的着法同样要过丢子守卫
                    mv, more = hang_guard(board, messages, mv, legal_sans, guard_cands,
                                          levels=check_levels, max_tokens=max_tokens)
                    guards += more
            if guards:
                obs["guard"] = guards
            if display_san(board, mv) != chosen_san:
                warnings.append(f"落子前复查后改选：{chosen_san} → {display_san(board, mv)}")
                obs["pv"] = []  # 原主变基于旧着法，已失效
            if PLAN_MEMORY and obs.get("strategy"):
                _plans[board.turn] = ([m.uci() for m in board.move_stack] + [mv.uci()], obs["strategy"])
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
