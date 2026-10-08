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
from datetime import datetime

import chess

from . import config
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


def lookup(board: chess.Board) -> dict | None:
    """当前局面在谱里就返回 {"uci","san","count","avg_cp","options"}：多个着法时取平均评估最高的
    （并列取次数多的），只考虑当前合法的着法；不在谱里返回 None。"""
    moves = _positions().get(board.epd())
    if not moves:
        return None
    best, best_key = None, None
    for uci, e in moves.items():
        try:
            if chess.Move.from_uci(uci) not in board.legal_moves:
                continue
        except ValueError:
            continue
        key = (e["cp_sum"] / e["count"], e["count"])
        if best_key is None or key > best_key:
            best_key = key
            best = {"uci": uci, "san": e["san"], "count": e["count"], "avg_cp": round(key[0]),
                    "options": len(moves)}
    return best


def keepers(evals: list[tuple[int, int]], floor_cp: int) -> list[tuple[int, int]]:
    """evals 是按顺序的 [(着法下标, 走完后我方 cp)]：保留第一个低于 floor_cp 的之前的那些。"""
    out = []
    for item in evals:
        if item[1] < floor_cp:
            break
        out.append(item)
    return out


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
                                              "count": 0, "cp_sum": 0}
                new += 1
            e["count"] += 1
            e["cp_sum"] += wanted[i]
            e["last"] = datetime.now().isoformat(timespec="seconds")
        board.push_uci(uci)
    return new


def commit_opening_book(uci_list: list[str], my_white: bool):
    """赛后调用：评估开局、把没变差的部分并入谱并存盘。"""
    if not config.BOOK_ENABLED or not uci_list:
        return
    evals = stockfish_opening_evals(uci_list, my_white, config.BOOK_MAX_MOVES, config.BOOK_FLOOR_CP)
    kept = keepers(evals, config.BOOK_FLOOR_CP)
    if not kept:
        print(f"[BOOK] 本局没有可入谱的着法（评估 {len(evals)} 步）")
        return
    new = record_line(uci_list, kept)
    _save()
    stopped = len(kept) < len(evals)
    print(f"[BOOK] 入谱 {len(kept)} 步（新增 {new}），"
          + (f"第 {len(kept) + 1} 步后评估低于 {config.BOOK_FLOOR_CP}cp，之后不入谱" if stopped else "未出现劣势")
          + f"；谱内共 {len(_positions())} 个局面")
