"""对局阶段的决策：复杂度分流 → LLM 选着（可调工具）→ 落子前自检 → 非法着法重试 / 保底。"""
import json
from concurrent.futures import ThreadPoolExecutor
import random

import chess

from . import board_view, book
from .archive import log_reasoning
from .boardtext import (COLOR_ZH, board_meta, describe_last_move, display_san, legal_san_map,
                        material_lead, parse_model_move, piece_lists, render_board, san_history,
                        strip_check)
from .config import (ANALYSIS_BOARD, BOARD_RELATIONS, BOOK_ENABLED, COMPLEXITY_CHECK,
                     COMPLEXITY_DEFAULT, COMPLEXITY_PROFILE, HANG_GUARD, HANG_GUARD_MIN, LLM_MAX_TOKENS,
                     MATERIAL_LEAD_EFFORT, MATERIAL_LEAD_SKIP, MATE_GUARD, OPENING_EFFORT, OPENING_FAST_MOVES,
                     PLAN_MEMORY, PV_FOLLOW_EFFORT, PV_MEMORY, SELF_CHECK_EFFORT, STRATEGY_STAGE,
                     STRATEGY_STAGE_MAX_TOKENS, TOOL_ROUNDS)
from .guard import allows_mate, mate_in_one, mate_threat, material_risk, risk_text
from .hooks import MoveContext, run_pre_move
from .live import live
from .llm import cap_ladder, extract_json, llm_call, reasoning_of, think_ladder
from .memory import recall_experience
from .prompts import (STRATEGY_STAGE_PROMPT, book_warning_section, complexity_prompt, prev_line_section,
                      strategy_stage_section, system_prompt, user_prompt)
from .tools import play_tools, run_tool


def fallback_move(legal_moves: list[str]) -> str:
    """模型多次给出非法走法时的最后保底：随机选一个合法着法，
    不做任何局面判断（杀棋 / 吃子 / 安全性都不替模型看），只保证不因超时或非法着法判负。"""
    return random.choice(legal_moves)


# 模型自己定下的计划：{我方颜色: {
#   "moves": 定下计划时的着法序列 UCI（含这一步我方着法）, "strategy": 战略方针,
#   "line": 主变 UCI（从这一步我方着法开始，程序按规则摆过，保证合法）, "goal": 主变的目的,
#   "mate": 主变摆到底是我方将杀}}。
# 新局面若不是在 moves 基础上继续（换了一盘棋），计划自动作废。
_plans: dict[bool, dict] = {}


def recall_plan(board: chess.Board) -> dict | None:
    stored = _plans.get(board.turn)
    if not stored:
        return None
    now = [m.uci() for m in board.move_stack]
    return stored if now[:len(stored["moves"])] == stored["moves"] else None


def line_from_pv(board: chess.Board, move: chess.Move, pv) -> tuple[list[str], bool]:
    """按规则摆一遍模型给的主变（pv[0] 必须是实际走的着法），在第一个摆不通的着法处截断。
    返回 (UCI 序列, 摆到底是否我方将杀)。"""
    if not isinstance(pv, list) or not pv or parse_model_move(board, str(pv[0])) != move:
        return [], False
    tmp = board.copy()
    line = []
    for text in pv:
        mv = parse_model_move(tmp, str(text))
        if mv is None:
            break
        line.append(mv.uci())
        tmp.push(mv)
    return line, len(line) % 2 == 1 and tmp.is_checkmate()


def line_progress(board: chess.Board, plan: dict | None) -> dict | None:
    """对照上一步保存的主变看对方刚走的一步。按主变应着时 next 为主变里我方的下一步，偏离时为 None；
    没有可用主变（没存、隔了不止一步、按主变应着但主变到此为止）返回 None。"""
    if not plan or len(plan["line"]) < 2 or len(board.move_stack) != len(plan["moves"]) + 1:
        return None
    before_reply = board.copy()
    reply = before_reply.pop()
    origin = before_reply.copy()
    origin.pop()
    line_san = []
    for u in plan["line"]:
        mv = chess.Move.from_uci(u)
        line_san.append(display_san(origin, mv))
        origin.push(mv)
    followed = reply.uci() == plan["line"][1]
    nxt = chess.Move.from_uci(plan["line"][2]) if followed and len(plan["line"]) > 2 else None
    if followed and (nxt is None or nxt not in board.legal_moves):
        return None
    return {"line_san": line_san, "expected": line_san[1], "actual": display_san(before_reply, reply),
            "next": nxt, "only_reply": before_reply.legal_moves.count() == 1}


def play_line_move(board: chess.Board, ply: int, plan: dict, prog: dict, why: str):
    """不调 LLM，直接走主变里我方的下一步，剩下的主变留给再下一步。"""
    mv = prog["next"]
    san = display_san(board, mv)
    _plans[board.turn] = dict(plan, moves=[m.uci() for m in board.move_stack] + [mv.uci()],
                              line=plan["line"][2:])
    note = f"按上一步主变直接走 {san}（{why}）。主变：{' '.join(prog['line_san'])}；目的：{plan['goal'] or '无'}"
    print(f"[PV] {note}")
    obs = {"strategy": plan["strategy"], "pv": prog["line_san"][2:], "pv_goal": plan["goal"],
           "line_follow": note}
    live.thinking(ply)
    live.decision(ply, move_san=san, move_uci=mv.uci(), think=note, opp_intent="", obs=obs,
                  reasoning="", recalled=[], warnings=[], attempts=0, fallback=False)
    return mv.uci(), note, "", obs


OBS_KEYS = ["complexity", "complexity_reason", "urgent", "strategy",
            "candidates", "pv", "pv_goal", "board_summary"]


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


def classify_complexity(board: chess.Board, last_section: str, legal_sans: list[str],
                        threat: bool = False) -> tuple[str, str]:
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
                               legal_sans=legal_sans, opp_sans=opp_sans, mate_threat=threat)
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


def safe_llm_move(board: chess.Board, ply: int, prev_board: chess.Board | None,
                  opp_last_move: str | None, **kw):
    """get_llm_move 的保底包装：LLM 接口异常（降档后仍失败）也不能让对局中断或超时，随机走一步合法着法。"""
    try:
        return get_llm_move(board, ply, prev_board, opp_last_move, **kw)
    except Exception as e:
        print(f"[ERROR] get_llm_move failed: {e!r}, random legal fallback")
        move = fallback_move([m.uci() for m in board.legal_moves])
        live.decision(ply, move_san=display_san(board, chess.Move.from_uci(move)), move_uci=move,
                      think="", opp_intent="", obs={},
                      warnings=[f"LLM 调用失败：{e}", "随机选择合法着法保底"], attempts=0, fallback=True)
        return move, "", "", {}


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

    # 一步杀（任一方）只用来分流：不背谱、不走快速模式、不直接续走主变；不告诉模型我方的杀着，
    # 对方的一步杀威胁只在 prompt 里提醒存在
    my_mate = MATE_GUARD and mate_in_one(board)
    opp_threat = MATE_GUARD and mate_threat(board)
    critical = my_mate or opp_threat
    if critical:
        print(f"[MATE] {'我方有一步杀 ' if my_mate else ''}{'对方有一步杀威胁' if opp_threat else ''}".strip())

    def safe_shortcut(mv: chess.Move) -> bool:
        """不经模型直接走 mv 是否稳妥：走完不会被一步杀；我方有一步杀时，mv 自己得是杀着。"""
        if not MATE_GUARD:
            return True
        if my_mate:
            after = board.copy(stack=False)
            after.push(mv)
            return after.is_checkmate()
        return not allows_mate(board, mv)

    # 上一步算出的主变：对方按主变应着时，连杀或对方唯一应着就不再推理，直接走下一步
    plan = recall_plan(board)
    prog = line_progress(board, plan) if PV_MEMORY else None
    followed = bool(prog and prog["next"] is not None)
    # 主变的下一步按吃子交换模拟会丢子（连杀主变除外，已按规则摆到将杀）：上一步的计算可能有误，
    # 不直接走、不降档，把模拟结果交给模型从当前盘面重新算
    next_hang = ""
    if followed and HANG_GUARD and not plan["mate"]:
        risk = material_risk(board, prog["next"])
        if risk["loss"] >= HANG_GUARD_MIN:
            next_hang = risk_text(board, prog["next"], risk)
            print(f"[PV] 对方按主变应着，但主变下一步会丢子，重新推理：{next_hang}")
    following = followed and not next_hang and safe_shortcut(prog["next"])
    if following:
        if plan["mate"]:
            return play_line_move(board, ply, plan, prog, "主变是连杀，对方按计算应着")
        if prog["only_reply"]:
            return play_line_move(board, ply, plan, prog, f"{prog['actual']} 是对方唯一的合法应着")

    # 背谱：当前局面在谱里时，按概率直接照走；否则（或抽到重新推理）走正常流程
    hit = book.lookup(board) if BOOK_ENABLED and not critical else None
    if hit:
        # 重新推理后又选了这一步的次数越多（n），越可信，越不必再花时间重复推理
        prob = book.play_prob(hit["n"], hit["avg_cp"])
        roll = random.random()
        if roll < prob:
            print(f"[BOOK] 背谱：{hit['san']}（确认 {hit['n']} 次，照走概率 {prob:.2f}；"
                  f"入谱 {hit['count']} 次，平均评估 {hit['avg_cp']:.0f}cp，该局面 {hit['options']} 个选择）")
            live.thinking(ply)
            note = f"背谱：{hit['san']}（确认 {hit['n']} 次，照走概率 {prob:.2f}，赛后评估平均 {hit['avg_cp']:.0f}cp）"
            obs = {"book": note}
            live.decision(ply, move_san=hit["san"], move_uci=hit["uci"], think=note, opp_intent="",
                          obs=obs, reasoning="", recalled=[], warnings=[], attempts=0, fallback=False)
            return hit["uci"], note, "", obs
        print(f"[BOOK] 局面在谱里（{hit['san']}，确认 {hit['n']} 次），抽到 {roll:.2f} ≥ {prob:.2f}，本步重新推理")

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
    fast = not critical and is_opening_fast(board, prev_board, opp_last_move)
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
    elif following:
        # 对方按主变应着：计划已经算过，只需确认，复杂度标签取起始档位相同的那一档
        complexity = next((k for k, (e, _) in COMPLEXITY_PROFILE.items() if e == PV_FOLLOW_EFFORT),
                          COMPLEXITY_DEFAULT)
        complexity_reason = f"对方按上一步主变应着，确认后续走 {display_san(board, prog['next'])}"
        ladder = think_ladder(PV_FOLLOW_EFFORT)
        max_tokens = COMPLEXITY_PROFILE.get(complexity, [None, LLM_MAX_TOKENS])[1]
        print(f"[PV] {complexity_reason} -> effort={PV_FOLLOW_EFFORT}, max_tokens={max_tokens}")
    elif not critical and MATERIAL_LEAD_SKIP > 0 and (lead := material_lead(board)) >= MATERIAL_LEAD_SKIP:
        # 子力大幅领先：不调 LLM 判断复杂度，直接用较低档位；复杂度标签取起始档位相同的那一档
        complexity = next((k for k, (e, _) in COMPLEXITY_PROFILE.items() if e == MATERIAL_LEAD_EFFORT),
                          COMPLEXITY_DEFAULT)
        complexity_reason = f"我方子力领先 {lead} 分，跳过复杂度判断"
        ladder = think_ladder(MATERIAL_LEAD_EFFORT)
        max_tokens = COMPLEXITY_PROFILE.get(complexity, [None, LLM_MAX_TOKENS])[1]
        print(f"[COMPLEXITY] {complexity_reason} -> effort={MATERIAL_LEAD_EFFORT}, max_tokens={max_tokens}")
    elif COMPLEXITY_CHECK:
        complexity, complexity_reason = classify_complexity(board, last_section, legal_sans, opp_threat)
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
        prev_strategy=(plan or {}).get("strategy", "") if PLAN_MEMORY else None,
        recalled=recall_section, legal_sans=legal_sans, fast=fast,
        complexity=complexity, complexity_reason=complexity_reason,
        prev_line=prev_line_section(prog["line_san"], plan["goal"], prog["expected"], prog["actual"],
                                    display_san(board, prog["next"]) if followed else "", next_hang)
        if prog else "",
        mate_threat=opp_threat)

    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": prompt},
    ]
    # 谱里记着这个局面下评分偏低的着法：先附上，第一阶段和决策阶段都能看到（开局快速模式也一样）
    book_warn = book.warnings(board) if BOOK_ENABLED else []
    if book_warn:
        messages[1]["content"] += book_warning_section(book_warn)
        print("[BOOK] 提醒：" + "；".join(f"{w['san']}（选过 {w['n']} 次，评分 {w['avg_cp'] / 100:+.2f}）"
                                          for w in book_warn))
    stage = strategy_stage(board, messages) if STRATEGY_STAGE and not fast and not following else None
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
            # 落子前 hook：丢子守卫 → 自检 → 将杀守卫（顺序与开关见 hooks.PRE_MOVE_HOOKS）
            ctx = MoveContext(
                board=board, messages=messages, legal_sans=legal_sans,
                candidates=[str(c.get("move", "")) for c in list(obs.get("candidates") or [])
                            + (stage or {}).get("candidates", []) if isinstance(c, dict)],
                complexity=str(obs.get("complexity", "")),
                check_levels=cap_ladder(ladder[step:], SELF_CHECK_EFFORT), levels=ladder[step:],
                max_tokens=max_tokens, fast=fast)
            mv, hook_records = run_pre_move(ctx, mv)
            obs.update(hook_records)
            if display_san(board, mv) != chosen_san:
                warnings.append(f"落子前复查后改选：{chosen_san} → {display_san(board, mv)}")
                obs["pv"] = []  # 原主变基于旧着法，已失效
            line, mate = line_from_pv(board, mv, obs.get("pv")) if PV_MEMORY else ([], False)
            # 模型这一步没写方针时沿用上一步的
            strategy = obs.get("strategy") or (plan or {}).get("strategy", "")
            if strategy or line:
                _plans[board.turn] = {"moves": [m.uci() for m in board.move_stack] + [mv.uci()],
                                      "strategy": strategy, "line": line, "mate": mate,
                                      "goal": str(obs.get("pv_goal") or "").strip()}
            if book_warn:
                obs["book_warning"] = [f"{w['san']}：选过 {w['n']} 次，评分 {w['avg_cp'] / 100:+.2f}"
                                       for w in book_warn]
            n_bad = next((w["n"] for w in book_warn if w["san"] == display_san(board, mv)), 0)
            if n_bad:
                warnings.append(f"选了谱里提醒过的 {display_san(board, mv)}（历史上选过 {n_bad} 次）")
            if BOOK_ENABLED:  # 重新推理的结果恰好是谱里已有的着法：确认次数 +1，下次更倾向直接背
                n = book.confirm(board, mv.uci())
                if n:
                    obs["book"] = f"重新推理后仍选谱里的 {display_san(board, mv)}，确认次数 {n}"
                    print(f"[BOOK] {obs['book']}")
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
