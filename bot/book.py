"""背谱：记住自己走过、且局面没有变差的开局，之后遇到同样局面按概率直接照走。

- 以局面（EPD：棋子分布 + 轮到谁 + 易位权 + 吃过路兵）为键，所以不同着法顺序走到同一局面也能命中，
  黑白双方的谱天然分开。
- 赛后（commit_opening_book）用 Stockfish 评估我方每步走完后的局面；第一次低于 BOOK_FLOOR_CP（默认 -1.0 兵）
  的那一步及之后都不入谱，之前的入谱。每个局面可以记多个着法，照走时选平均评估最高的。
- 对局中（lookup）命中后由 player 按 BOOK_PLAY_PROB 决定直接照走还是重新推理；
  走到谱里没有的局面后就一直正常推理。
"""
import json
import os
import random
from datetime import datetime

import chess

from . import config
from .boardtext import san_history
from .engine import stockfish_opening_evals

_book: dict | None = None


def _positions() -> dict:
    """{局面 EPD: {着法 UCI: {san, count, cp_sum, last}}}，首次使用时从文件读入。"""
    global _book
    if _book is None:
        _book = {}
        try:
            with open(config.BOOK_PATH, encoding="utf-8") as f:
                _book = json.load(f).get("positions", {})
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[BOOK] 读取 {config.BOOK_PATH} 失败，当作空谱: {e}")
    return _book


def _save():
    os.makedirs(os.path.dirname(config.BOOK_PATH) or ".", exist_ok=True)
    tmp = config.BOOK_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "positions": _positions()}, f, ensure_ascii=False, indent=1)
    os.replace(tmp, config.BOOK_PATH)


def _entries(board: chess.Board) -> list[dict]:
    """当前局面在谱里记过的、现在合法的着法，每项 {"uci","san","count","avg_cp","n"}。"""
    out = []
    for uci, e in _positions().get(board.epd(), {}).items():
        try:
            if chess.Move.from_uci(uci) not in board.legal_moves:
                continue
        except ValueError:
            continue
        out.append({"uci": uci, "san": e["san"], "count": e["count"], "avg_cp": e["cp_sum"] / e["count"],
                    "n": 1 + e.get("confirm", 0)})
    return out


def lookup(board: chess.Board) -> dict | None:
    """当前局面在谱里有可背的着法就返回 {"uci","san","count","avg_cp","n","options"}：
    必定取赛后评估平均最高的，评估相同（差不到 0.01 兵）就在它们之中随机选一个。
    评估平均低于 BOOK_FLOOR_CP 的着法是"错棋记录"，只用来提醒，不会被背；没有可背的返回 None。"""
    entries = _entries(board)
    playable = [e for e in entries if e["avg_cp"] >= config.BOOK_FLOOR_CP]
    if not playable:
        return None
    top = max(e["avg_cp"] for e in playable)
    best = random.choice([e for e in playable if top - e["avg_cp"] < 1])
    return dict(best, options=len(entries))


def warnings(board: chess.Board, k: int = 3) -> list[dict]:
    """当前局面谱里评估平均为负的着法中最低的 k 个（含不会被背的错棋记录），按评分从低到高。"""
    bad = [e for e in _entries(board) if e["avg_cp"] < 0]
    return sorted(bad, key=lambda e: e["avg_cp"])[:k]


def play_prob(n: int, avg_cp: float = 0) -> float:
    """命中谱时直接照走的概率。n = 这一步被选中的次数（入谱算 1 次，之后每次重新推理又选了它加 1）。
    - 赛后评估平均 ≥ 0：BOOK_PLAY_PROB + (1 - BOOK_PLAY_PROB) * (1 - 1/n)，
      n=1 即基础概率（默认 0.6），n 越大越趋近 1，省得对反复确认的着法重复推理；
    - -1.0 兵 ≤ 平均 < 0（这步让我方略处下风）：BOOK_NEG_PLAY_PROB / (n + 1)（默认 0.2），
      越是反复选到越少背，错棋尽量少选；
    - 平均 < BOOK_FLOOR_CP：0，只做提醒，不背。"""
    n = max(1, n)
    if avg_cp < config.BOOK_FLOOR_CP:
        return 0.0
    if avg_cp < 0:
        return config.BOOK_NEG_PLAY_PROB / (n + 1)
    base = config.BOOK_PLAY_PROB
    return base + (1 - base) * (1 - 1 / n)


def confirm(board: chess.Board, uci: str) -> int:
    """局面在谱里、模型重新推理后选了谱里已有的 uci：确认次数 +1 并存盘，返回新的 n；
    该着法不在谱里（会在赛后评估通过后作为新选择入谱）返回 0。"""
    e = _positions().get(board.epd(), {}).get(uci)
    if e is None:
        return 0
    e["confirm"] = e.get("confirm", 0) + 1
    _save()
    return 1 + e["confirm"]


def keepers(evals: list[tuple[int, int]], floor_cp: int) -> list[tuple[int, int]]:
    """evals 是按顺序的 [(着法下标, 走完后我方 cp)]：保留第一个低于 floor_cp 的之前的那些。"""
    out = []
    for item in evals:
        if item[1] < floor_cp:
            break
        out.append(item)
    return out


def faulty(evals: list[tuple[int, int]], floor_cp: int) -> tuple[int, int] | None:
    """第一个评估低于 floor_cp 的着法（让局面由可接受变成劣势的那一步），没有则 None。"""
    return next((item for item in evals if item[1] < floor_cp), None)


def record_line(uci_list: list[str], kept: list[tuple[int, int]]) -> int:
    """把 kept 里的着法并入谱（局面取走这步之前的），返回新增的（局面, 着法）条数。"""
    book = _positions()
    wanted = {i: cp for i, cp in kept}
    board = chess.Board()
    new = 0
    for i, uci in enumerate(uci_list):
        if i in wanted:
            e = book.setdefault(board.epd(), {}).get(uci)
            if e is None:
                e = book[board.epd()][uci] = {"san": board.san(chess.Move.from_uci(uci)),
                                              "via": san_history(board),  # 首次走到该局面的着法顺序，仅供查看
                                              "count": 0, "cp_sum": 0}
                new += 1
            e["count"] += 1
            e["cp_sum"] += wanted[i]
            e["last"] = datetime.now().isoformat(timespec="seconds")
        board.push_uci(uci)
    return new


def commit_opening_book(uci_list: list[str], my_white: bool):
    """赛后调用：评估开局，没变差的部分并入谱；第一个让评估低于下限的那步也记下来（只做提醒，不会被背），
    之后的着法不记。存盘。"""
    if not config.BOOK_ENABLED or not uci_list:
        return
    evals = stockfish_opening_evals(uci_list, my_white, config.BOOK_MAX_MOVES, config.BOOK_FLOOR_CP)
    kept = keepers(evals, config.BOOK_FLOOR_CP)
    bad = faulty(evals, config.BOOK_FLOOR_CP)
    if not kept and not bad:
        print(f"[BOOK] 本局没有可记录的着法（评估 {len(evals)} 步）")
        return
    new = record_line(uci_list, kept + ([bad] if bad else []))
    _save()
    print(f"[BOOK] 入谱 {len(kept)} 步（新增 {new}），"
          + (f"第 {len(kept) + 1} 步评估 {bad[1]}cp 低于 {config.BOOK_FLOOR_CP}cp，记为错棋提醒，之后不记"
             if bad else "未出现劣势")
          + f"；谱内共 {len(_positions())} 个局面")
