"""落子前 hook：模型选出着法后、真正落子前依次执行的检查（自检、丢子守卫、将杀守卫）。

每个 hook 接收当前着法，返回 (最终着法, 本 hook 的记录)，可以放行、要求模型改选或直接替换。
执行顺序和开关都在 PRE_MOVE_HOOKS 里登记；以后加新守卫（捉双、牵制……）只需写一个函数并登记：
- recheck：规则类守卫。后面的 hook 改了着法时，新着法要再过一遍它（自检改出来的着法同样要过丢子守卫）。
- final：最后的裁决。它改出来的着法不再触发 recheck（被一步杀压过一切）。
"""
from dataclasses import dataclass
from typing import Callable

import chess

from . import board_view
from .archive import log_reasoning
from .boardtext import COLOR_ZH, display_san, parse_model_move, render_board
from .config import (BOARD_RELATIONS, HANG_GUARD, HANG_GUARD_MIN, HANG_GUARD_POSITIONAL, HANG_GUARD_ROUNDS,
                     MATE_GUARD, MATE_GUARD_ROUNDS, SELF_CHECK_ROUNDS)
from .guard import allows_mate, material_risk, risk_text, verify_line
from .live import live
from .llm import extract_json, llm_call, reasoning_of
from .prompts import hang_guard_pick_prompt, hang_guard_prompt, mate_guard_prompt, self_check_prompt


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
    轮数用完仍未解决时走 _guard_fallback：候选里不丢子的 → 全部合法着法里不丢子的 → 净亏最小的。
    每条记录带 outcome（kept_tactical / kept_positional / changed / fallback / fallback_pick /
    fallback_legal / least_loss / unresolved），供赛后统计。"""
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
    risk = material_risk(board, current)
    if risk["loss"] < HANG_GUARD_MIN:
        return current, records
    # current 可能是最后一轮改出来、从没被复查过的着法，不能当作"原着法"保持
    rejected[current] = risk_text(board, current, risk)
    mv, outcome, note = _guard_fallback(board, messages, current, candidates, rejected, levels, max_tokens)
    print(f"[GUARD] {note}")
    records.append({"round": len(records) + 1, "checked": display_san(board, current),
                    "checked_uci": current.uci(), "outcome": outcome, "note": note,
                    **({"changed_to": display_san(board, mv)} if mv != current else {})})
    return mv, records


def _guard_fallback(board: chess.Board, messages: list, current: chess.Move, candidates: list[str],
                    rejected: dict[chess.Move, str], levels: list[str] | None,
                    max_tokens: int | None) -> tuple[chess.Move, str, str]:
    """丢子守卫轮数用完仍会丢子时的兜底，返回 (着法, outcome, note)：
    1. 模型候选里第一个不丢子的；
    2. 全部合法着法里按规则不丢子的：交给模型在这份名单里挑一个，挑不出就取净亏最小的；
    3. 都会丢子：取全部合法着法里净亏最小的（同分时优先模型提过的着法）。"""
    loss = {m: material_risk(board, m)["loss"] for m in board.legal_moves}
    for text in candidates:
        mv = parse_model_move(board, text)
        if mv is not None and mv not in rejected and loss[mv] < HANG_GUARD_MIN:
            return mv, "fallback", f"复查后仍未解决，改用候选里不丢子的 {display_san(board, mv)}"
    safe = sorted((m for m in loss if loss[m] < HANG_GUARD_MIN and m not in rejected), key=loss.get)
    if safe:
        safe_sans = [display_san(board, m) for m in safe]
        print(f"[GUARD] 候选都会丢子，按规则不丢子的合法着法：{', '.join(safe_sans)}")
        live.stage("丢子守卫：从不丢子的着法里改选")
        prompt = hang_guard_pick_prompt({display_san(board, m): r for m, r in rejected.items()}, safe_sans)
        pick = None
        try:
            msg, _ = llm_call(messages + [{"role": "user", "content": prompt}], levels=levels,
                              max_tokens=max_tokens)
            content = (msg.content or "").strip()
            print(f"[GUARD pick] {content}")
            log_reasoning("GUARD pick", reasoning_of(msg))
            pick = parse_model_move(board, str((extract_json(content) or {}).get("move", "")))
        except Exception as e:
            print(f"[GUARD] LLM failed: {e}")
        if pick in safe:
            return pick, "fallback_pick", f"复查后仍未解决，从不丢子的合法着法里改选 {display_san(board, pick)}"
        return safe[0], "fallback_legal", f"复查后仍未解决，改用合法着法里不丢子的 {display_san(board, safe[0])}"
    proposed = set(rejected) | {m for m in (parse_model_move(board, c) for c in candidates) if m is not None}
    mv = min(loss, key=lambda m: (loss[m], m not in proposed))
    if mv == current:
        return mv, "unresolved", f"所有合法着法都会丢子，{display_san(board, mv)} 已是净亏最小的"
    return mv, "least_loss", (f"所有合法着法都会丢子，改用净亏最小的 {display_san(board, mv)}"
                              f"（净亏 {loss[mv]}，{display_san(board, current)} 净亏 {loss[current]}）")


def mate_guard(board: chess.Board, messages: list, move: chess.Move, legal_sans: list[str],
               candidates: list[str], levels: list[str] | None = None,
               max_tokens: int | None = None) -> tuple[chess.Move, list[dict]]:
    """将杀守卫：程序按规则查到走完 move 后对方有一步杀时，只告诉模型"存在一步杀"（不给对方着法），
    要求改选；被将杀没有补偿可言，不允许坚持。改选的着法同样要查，最多 MATE_GUARD_ROUNDS 轮。
    仍未解决时依次从模型的候选、全部合法着法里找不会被一步杀的（优先不丢子的），都没有则保持原着法。"""
    records: list[dict] = []
    current = move
    rejected: list[chess.Move] = []  # 查到会被一步杀的着法
    for rnd in range(1, MATE_GUARD_ROUNDS + 1):
        if not allows_mate(board, current):
            return current, records
        san = display_san(board, current)
        rejected.append(current)
        print(f"[MATE-GUARD round={rnd}] 走完 {san} 后对方有一步杀")
        live.stage(f"将杀守卫：{san} 会被一步杀")
        prompt = mate_guard_prompt(san, legal_sans, [display_san(board, m) for m in rejected[:-1]])
        try:
            msg, _ = llm_call(messages + [{"role": "user", "content": prompt}], levels=levels,
                              max_tokens=max_tokens)
        except Exception as e:
            print(f"[MATE-GUARD] LLM failed: {e}")
            break
        content = (msg.content or "").strip()
        print(f"[MATE-GUARD reply round={rnd}] {content}")
        log_reasoning(f"MATE-GUARD round={rnd}", reasoning_of(msg))
        obj = extract_json(content) or {}
        rec = {"round": rnd, "checked": san, **{k: str(obj.get(k, "")).strip() for k in ("move", "reason")},
               "reasoning": reasoning_of(msg)}
        records.append(rec)
        new = parse_model_move(board, rec["move"])
        if new is None or new in rejected:
            rec["note"] = f"改选的着法无效或同样会被一步杀：{rec['move']!r}"
            print(f"[MATE-GUARD] {rec['note']}")
            break
        rec["outcome"], rec["changed_to"] = "changed", display_san(board, new)
        print(f"[MATE-GUARD] 改选 {san} -> {rec['changed_to']}")
        current = new
    if not allows_mate(board, current):
        return current, records
    pool = [m for m in (parse_model_move(board, c) for c in candidates) if m is not None] \
        + list(board.legal_moves)
    safe = [m for m in dict.fromkeys(pool) if m != current and not allows_mate(board, m)]
    if safe:
        mv = next((m for m in safe if material_risk(board, m)["loss"] < HANG_GUARD_MIN), safe[0])
        outcome, note = "fallback", f"复查后仍会被一步杀，改用不会被一步杀的 {display_san(board, mv)}"
    else:
        mv, outcome, note = current, "unresolved", "所有合法着法都会被一步杀，保持原着法"
    print(f"[MATE-GUARD] {note}")
    records.append({"round": len(records) + 1, "checked": display_san(board, current), "outcome": outcome,
                    "note": note, **({"changed_to": display_san(board, mv)} if mv != current else {})})
    return mv, records



@dataclass
class MoveContext:
    """一步决策里 hook 需要的上下文。check_levels：复查用的档位（已按 SELF_CHECK_EFFORT 封顶）；
    levels：本步决策的档位（不封顶，将杀守卫要在防法里算出最强的）。"""
    board: chess.Board
    messages: list
    legal_sans: list[str]
    candidates: list[str]
    complexity: str
    check_levels: list[str]
    levels: list[str]
    max_tokens: int | None
    fast: bool = False


@dataclass
class Hook:
    name: str
    obs_key: str  # 记录写进 obs 的哪个键（观战页面与赛后统计按这个键读）
    enabled: Callable[[MoveContext], bool]
    run: Callable[[MoveContext, chess.Move], tuple[chess.Move, list[dict]]]
    recheck: bool = False
    final: bool = False


PRE_MOVE_HOOKS = [
    Hook("hang_guard", "guard", lambda ctx: HANG_GUARD,
         lambda ctx, mv: hang_guard(ctx.board, ctx.messages, mv, ctx.legal_sans, ctx.candidates,
                                    levels=ctx.check_levels, max_tokens=ctx.max_tokens),
         recheck=True),
    Hook("self_check", "self_check", lambda ctx: SELF_CHECK_ROUNDS > 0 and not ctx.fast,
         lambda ctx, mv: self_check(ctx.board, ctx.messages, mv, ctx.legal_sans, ctx.complexity,
                                    levels=ctx.check_levels, max_tokens=ctx.max_tokens)),
    Hook("mate_guard", "mate_guard", lambda ctx: MATE_GUARD,
         lambda ctx, mv: mate_guard(ctx.board, ctx.messages, mv, ctx.legal_sans, ctx.candidates,
                                    levels=ctx.levels, max_tokens=ctx.max_tokens),
         final=True),
]


def run_pre_move(ctx: MoveContext, move: chess.Move,
                 hooks: list[Hook] | None = None) -> tuple[chess.Move, dict[str, list[dict]]]:
    """按顺序执行启用的 hook，返回 (最终着法, {obs_key: 记录})，只含有记录的键。"""
    hooks = PRE_MOVE_HOOKS if hooks is None else hooks
    records: dict[str, list[dict]] = {}

    def run(hook: Hook, mv: chess.Move) -> chess.Move:
        new, recs = hook.run(ctx, mv)
        if recs:
            records.setdefault(hook.obs_key, []).extend(recs)
        return new

    for i, hook in enumerate(hooks):
        if not hook.enabled(ctx):
            continue
        before = move
        move = run(hook, move)
        if move != before and not hook.final:
            for prev in hooks[:i]:
                if prev.recheck and prev.enabled(ctx):
                    move = run(prev, move)
    return move, records
