"""赛后复盘：自我反思、Stockfish blunder 深挖、聊天总结、局面快照验证，结果写入经验库。"""
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import chess

from . import skills
from .boardtext import parse_model_move, san_history, uci_to_san
from .book import commit_opening_book
from .config import MODEL, REVIEW_WORKERS, SNAPSHOT_DEDUPE_SIM, SNAPSHOT_OK_DELTA
from .engine import stockfish_collect_blunders, stockfish_eval_move
from .llm import complete, extract_json
from .memory import BAD_VERDICTS, add_lesson, experience_rag
from .tools import REVIEW_TOOLS, run_tool


def post_game_review(pgn_text: str, result: str, my_color: str, move_log: list):
    print("[REVIEW] generating self-review ...")
    brief_moves = "\n".join(
        f"ply{p}. {uci_to_san(fen, mv)}  思考:{(th or '')[:40]}"
        for p, fen, mv, th in move_log[-40:]
    )
    system = """你正在复盘一盘刚下完的国际象棋。流程：
1. 先自行思考这盘棋的大致表现。
2. 如果你想验证具体哪些步走错了、引擎评估是多少，可调用 analyze_with_stockfish 工具（仅此时可用）。
3. 可选地调用 search_experience 看看是否和过往教训重合。
4. 最后严格输出 JSON（不要 markdown）：
{
  "summary": "总体不超过 150 字",
  "lessons": ["教训1（详细分析，≤100字）", "教训2", "教训3"]
}
每条 lesson 必须详细描述「局面特征 + 失误/正确决策 + 应对原则」三要素，便于未来向量检索复用，单条 ≤100 字，可以写到接近 100 字。"""

    user = f"""你执{my_color}，结果：{result}
PGN:
{pgn_text}

你每步的简短思考（最后 40 步）：
{brief_moves}
"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    ctx = {"pgn_text": pgn_text, "my_color": my_color}

    text = ""
    try:
        for _ in range(3):
            resp = complete(
                model=MODEL,
                messages=messages,
                tools=REVIEW_TOOLS,
                temperature=0.3,
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                })
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    result_tool = run_tool(tc.function.name, args, ctx=ctx)
                    print(f"[REVIEW-TOOL] {tc.function.name}({args}) -> {result_tool[:160]}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_tool,
                    })
                continue
            text = (msg.content or "").strip()
            break
    except Exception as e:
        print(f"[REVIEW] LLM call failed: {e}")
        return

    obj = extract_json(text)
    if obj is None:
        print(f"[REVIEW] no JSON parsed. raw={text[:200]}")
        return

    summary = obj.get("summary", "")
    lessons = obj.get("lessons", [])
    print(f"[REVIEW] summary: {summary}")
    meta = {"result": result, "color": my_color,
            "time": datetime.now().isoformat()}
    for ls in lessons:
        if isinstance(ls, str) and ls.strip():
            print(f"[REVIEW] +lesson ({add_lesson(ls.strip(), meta)}): {ls}")
    experience_rag.add(f"[复盘] {summary}", meta)
    print(f"[REVIEW] experience size = {len(experience_rag)}")


def _analyze_blunder(i: int, n: int, b: dict) -> dict | None:
    """单个 blunder：让模型分析失误并给出替代着法，再用 Stockfish 评估替代着法。
    各 blunder 互不依赖，blunder_deep_review 并行调用；这里不写库，由调用方按顺序写入。"""
    tag = f"[BLUNDER {i}/{n}]"
    print(f"\n{tag} ply={b['ply']} {b['side']} 走了 {b['san']} (best={b['best_san']}) "
          f"cp {b['cp_before']} -> {b['cp_after']} (Δ={b['delta']})")

    prompt = f"""下面是本局中一个被 Stockfish 标记为 blunder 的关键节点。

阵营: {b['side']} 走子
回合(ply): {b['ply']}
此前的全部着法（SAN）: {b['history'] or '（开局）'}
实际走法: {b['san']}
引擎推荐: {b['best_san']}
评估变化(走子方视角, cp): {b['cp_before']} -> {b['cp_after']}  (Δ={b['delta']})

走子前盘面(白=W*, 黑=B*, '.'=空):
{b['board_before']}

走子后盘面:
{b['board_after']}

请：
1. 对比两个盘面的差异（哪些子动了、丢了什么、暴露了什么）。
2. 分析为什么这步是 blunder：忽视了什么威胁/战术？王安全/子力/兵形上有何问题？
3. 给出你认为「在走子前的局面下」更好的着法（your_better_move，SAN 记谱，如 Nf3），并简要说明理由。
4. 提取一条可复用的经验（lesson，描述局面特征 + 失误模式 + 应对原则，≤100 字）。

严格输出 JSON（不要 markdown）：
{{
  "diff": "≤80 字",
  "why_blunder": "≤120 字",
  "your_better_move": "e4",
  "your_reason": "≤80 字",
  "lesson": "≤100 字"
}}"""
    try:
        resp = complete(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        txt = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"{tag} LLM failed: {e}")
        return None

    obj = extract_json(txt)
    if obj is None:
        print(f"{tag} no JSON. raw={txt[:200]}")
        return None

    my_better = (obj.get("your_better_move") or "").strip()
    print(f"{tag} diff: {obj.get('diff', '')}")
    print(f"{tag} why : {obj.get('why_blunder', '')}")
    print(f"{tag} mine: {my_better}  reason: {obj.get('your_reason', '')}")

    # 用 Stockfish 评估模型给出的"更好走法"
    alt_mv = parse_model_move(chess.Board(b["fen_before"]), my_better) if my_better else None
    sf_eval = stockfish_eval_move(b["fen_before"], alt_mv.uci()) if alt_mv else \
        {"error": f"无法解析或非法的着法: {my_better!r}"}
    print(f"{tag} sf_eval: {sf_eval}")
    return {"better": my_better, "reason": obj.get("your_reason", ""),
            "lesson": (obj.get("lesson") or "").strip(), "sf_eval": sf_eval}


def blunder_deep_review(pgn_text: str, result: str, my_color: str) -> list[dict]:
    """强制：用 Stockfish 找出本局所有 blunder（双方），
    对每个 blunder 让模型分析两个 FEN 的差异并给出新判断，
    再用 Stockfish 评估这个新判断，全部写入经验库。
    各 blunder 的分析并行（REVIEW_WORKERS 个线程），写库按 blunder 顺序。返回找到的 blunder。"""
    print("[BLUNDER-REVIEW] start ...")
    blunders = stockfish_collect_blunders(pgn_text, threshold_cp=200)
    if not blunders:
        print("[BLUNDER-REVIEW] no blunder found")
        return []
    print(f"[BLUNDER-REVIEW] {len(blunders)} blunder(s) found")

    meta_base = {"result": result, "my_color": my_color,
                 "kind": "blunder", "time": datetime.now().isoformat()}

    n = len(blunders)
    with ThreadPoolExecutor(max_workers=max(1, REVIEW_WORKERS)) as pool:
        analyses = list(pool.map(lambda ib: _analyze_blunder(ib[0], n, ib[1]), enumerate(blunders, 1)))

    for i, (b, a) in enumerate(zip(blunders, analyses), 1):
        if a is None:
            continue
        sf_eval = a["sf_eval"]
        evaluated = isinstance(sf_eval, dict) and "error" not in sf_eval
        verdict = sf_eval.get("verdict", "?") if evaluated else ""
        meta = dict(meta_base)
        meta.update({"ply": b["ply"], "side": b["side"],
                     "actual_move": b["san"], "best": b["best_san"],
                     "delta": b["delta"]})
        if verdict:
            meta["alt_verdict"] = verdict

        # 写入主 lesson。模型据此给出的替代着法被 Stockfish 判为 mistake / blunder 时，
        # 这条"应对原则"已被证伪，不写入（替代走法评估照常记录）
        if a["lesson"] and verdict in BAD_VERDICTS:
            print(f"[BLUNDER {i}] lesson 不写入：替代着法 {a['better']} 被 Stockfish 判为 {verdict}")
        elif a["lesson"]:
            entry = (f"[Blunder-Lesson] {a['lesson']} "
                     f"(局面: ply{b['ply']} {b['side']}方走 {b['san']}, "
                     f"引擎推荐 {b['best_san']}, Δ={b['delta']}cp)")
            print(f"[BLUNDER {i}] +lesson ({add_lesson(entry, meta)}): {entry[:120]}")

        # 写入模型对自身替代走法的评估
        if evaluated:
            entry2 = (
                f"[Blunder-AltMove] 走法历史: {b['history'] or '（开局）'} | "
                f"模型替代走法 {a['better']} 理由: {a['reason']} | "
                f"引擎评估: cp {sf_eval['cp_before']}->{sf_eval['cp_after']} "
                f"(Δ={sf_eval['delta']}, {verdict}); "
                f"引擎最佳 {uci_to_san(b['fen_before'], sf_eval['best'])}. "
                f"原失误走法 {b['san']} Δ={b['delta']}cp."
            )
            print(f"[BLUNDER {i}] +alt-eval: {entry2[:120]}")
            experience_rag.add(entry2, dict(meta, kind="blunder_alt_eval"))
        else:
            print(f"[BLUNDER {i}] alt eval skipped: {sf_eval}")

    print(f"[BLUNDER-REVIEW] done. experience size = {len(experience_rag)}")
    return blunders


def chat_review(chat_messages: list, result: str, my_color: str, my_username: str):
    """把本局聊天框中收到的指导整理为经验。读取 only。"""
    if not chat_messages:
        return
    # 过滤掉自己（理论上 bot 不发，但保险）
    lines = [c for c in chat_messages
             if c.get("username", "").lower() != my_username.lower()]
    if not lines:
        return
    print(f"[CHAT-REVIEW] {len(lines)} message(s) to summarize")
    joined = "\n".join(
        f"[ply={c.get('ply','?')} 此前着法: {c.get('history') or '（开局）'}]\n"
        f"[{c.get('room','player')}] {c.get('username','')}: {c.get('text','')}"
        for c in lines
    )
    prompt = f"""本局({my_color}方, 结果 {result})期间收到的聊天指导（按时间顺序）：

{joined}

请从中提炼可复用的国际象棋经验/原则。
严格输出 JSON（不要 markdown）：
{{
  "summary": "≤120 字，本局教练给的整体方向",
  "lessons": ["≤100 字一条", "..."]
}}
lessons 数量 1~5 条；只保留有普适价值的内容，闲聊忽略。"""
    try:
        resp = complete(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        txt = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[CHAT-REVIEW] LLM failed: {e}")
        return
    obj = extract_json(txt)
    if obj is None:
        print(f"[CHAT-REVIEW] no JSON. raw={txt[:200]}")
        return
    summary = obj.get("summary", "").strip()
    lessons = obj.get("lessons", []) or []
    meta = {"result": result, "my_color": my_color,
            "kind": "chat_guidance", "time": datetime.now().isoformat()}
    if summary:
        experience_rag.add(f"[Chat-Summary] {summary}", meta)
        print(f"[CHAT-REVIEW] +summary: {summary[:120]}")
    for ls in lessons:
        if isinstance(ls, str) and ls.strip():
            status = add_lesson(f"[Chat-Lesson] {ls.strip()}", meta)
            print(f"[CHAT-REVIEW] +lesson ({status}): {ls.strip()[:120]}")
    # 同时把原始聊天存档供未来回溯
    experience_rag.add(
        f"[Chat-Raw] {joined[:800]}",
        dict(meta, kind="chat_raw"),
    )
    print(f"[CHAT-REVIEW] done. experience size = {len(experience_rag)}")


def record_snapshot(snapshots: list, board: chess.Board, move: str, obs: dict):
    """对局中缓存一条快照（走子前 FEN + 模型自己的摘要/PV），不写库。"""
    bs = (obs.get("board_summary") or "").strip() if obs else ""
    if not bs:
        return
    pv = obs.get("pv") or []
    snapshots.append({
        "fen": board.fen(),  # 仅供赛后 Stockfish 验证，不写入经验库文本
        "history": san_history(board),
        "ply": board.ply() + 1,
        "move": move,
        "san": uci_to_san(board.fen(), move),
        "summary": bs,
        "pv": " ".join(map(str, pv)) if isinstance(pv, list) else str(pv),
    })


def commit_verified_snapshots(snapshots: list, result: str, my_color: str):
    """赛后用 Stockfish 验证每条快照里模型选的着法，只有"好棋"才写入经验库，
    避免把模型自己的错误判断当成经验反复召回。同时对相近条目去重。"""
    if not snapshots:
        return
    kept = skipped_bad = skipped_dup = 0
    # 每次评估各自起一个 Stockfish 进程，可以并行；写库按原顺序
    with ThreadPoolExecutor(max_workers=max(1, REVIEW_WORKERS)) as pool:
        evals = list(pool.map(lambda s: stockfish_eval_move(s["fen"], s["move"]), snapshots))
    for s, ev in zip(snapshots, evals):
        if "error" in ev or ev.get("delta", 10**9) >= SNAPSHOT_OK_DELTA:
            skipped_bad += 1
            continue
        entry = (f"[Verified-Snapshot] 走法历史: {s['history'] or '（开局）'} | "
                 f"摘要: {s['summary']} | "
                 f"选择: {s['san']}（Stockfish 验证 Δ={ev['delta']}cp，"
                 f"引擎最佳 {uci_to_san(s['fen'], ev['best'])}） | PV: {s['pv']}")
        wrote = experience_rag.add(entry, {
            "kind": "verified_snapshot", "verified": True,
            "ply": s["ply"], "result": result, "color": my_color,
            "delta": ev["delta"], "time": datetime.now().isoformat(),
        }, dedupe_threshold=SNAPSHOT_DEDUPE_SIM)
        if wrote:
            kept += 1
        else:
            skipped_dup += 1
    print(f"[SNAP] {len(snapshots)} 条快照: 写入 {kept}, 未通过验证 {skipped_bad}, 重复 {skipped_dup}")


def run_post_game(pgn_text: str, result: str, my_color: str, move_log: list, snapshots: list,
                  uci_list: list[str], my_white: bool, chat_messages: list | None = None,
                  my_username: str = ""):
    """赛后各项复盘互不依赖（各自调 LLM、各自起 Stockfish，写库有锁），并行跑；某一项失败不影响其他项。"""
    tasks = {
        "REVIEW": lambda: post_game_review(pgn_text, result, my_color, move_log),
        # 技能统计要用 blunder 列表：本局哪些步命中了哪些技能、其中几步是 blunder
        "BLUNDER-REVIEW": lambda: skills.record_game(pgn_text, my_color,
                                                     blunder_deep_review(pgn_text, result, my_color)),
        "SNAP": lambda: commit_verified_snapshots(snapshots, result, my_color),
        "BOOK": lambda: commit_opening_book(uci_list, my_white),
    }
    if chat_messages:
        tasks["CHAT-REVIEW"] = lambda: chat_review(chat_messages, result, my_color, my_username)
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {tag: pool.submit(fn) for tag, fn in tasks.items()}
    for tag, fut in futures.items():
        if fut.exception() is not None:
            print(f"[{tag}] failed: {fut.exception()!r}")
