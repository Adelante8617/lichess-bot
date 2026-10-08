"""Stockfish 分析：仅用于赛后复盘，下棋阶段不调用。"""
import io
import json

import chess
import chess.engine
import chess.pgn

from .boardtext import render_board, san_history
from .config import STOCKFISH_ANALYZE_DEPTH, STOCKFISH_PATH

CP_CLAMP = 1000  # 评估值截断：避免 mate 被换算成天文数字，让 delta 失真


def _cp_from_score(score: chess.engine.PovScore, pov_white: bool) -> int:
    """统一取指定视角的 centipawn，mate 与超大评估都截断到 ±CP_CLAMP。"""
    s = score.white() if pov_white else score.black()
    cp = s.score(mate_score=10000)
    if cp is None:
        return 0
    return max(-CP_CLAMP, min(CP_CLAMP, int(cp)))


def _classify(delta_cp: int) -> str:
    # delta_cp: 走完这步后己方失去了多少（正数 = 变差）
    if delta_cp >= 300:
        return "blunder"
    if delta_cp >= 100:
        return "mistake"
    if delta_cp >= 50:
        return "inaccuracy"
    return "ok"


def stockfish_analyze_pgn(pgn_text: str, my_color: str, max_points: int = 8) -> str:
    """对 PGN 做 Stockfish 分析，返回本方关键失误节点的 JSON 字符串。"""
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        msg = f"Stockfish 启动失败（请确认 STOCKFISH_PATH 或把 stockfish 放入 PATH）: {e}"
        print(f"[STOCKFISH] {msg}")
        return json.dumps({"error": msg}, ensure_ascii=False)

    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return json.dumps({"error": "parse pgn failed"}, ensure_ascii=False)

        board = game.board()
        my_is_white = (my_color == "白")
        limit = chess.engine.Limit(depth=STOCKFISH_ANALYZE_DEPTH)

        # 走子前先评估，再走一步再评估，计算本方视角的 cp 损失
        points = []
        ply = 0
        for move in game.mainline_moves():
            ply += 1
            mover_is_white = (board.turn == chess.WHITE)
            mover_is_me = (mover_is_white == my_is_white)

            try:
                info_before = engine.analyse(board, limit)
                # 我们从「刚走完此手方」的视角看 cp 变化
                cp_before = _cp_from_score(info_before["score"], pov_white=mover_is_white)
                best = info_before.get("pv", [None])[0]
                best_uci = best.uci() if best else ""
            except Exception as e:
                print(f"[STOCKFISH] analyse before failed: {e}")
                break

            board.push(move)

            try:
                info_after = engine.analyse(board, limit)
                cp_after = _cp_from_score(info_after["score"], pov_white=mover_is_white)
            except Exception as e:
                print(f"[STOCKFISH] analyse after failed: {e}")
                break

            delta = cp_before - cp_after  # 本方损失（正数越大越差）
            tag = _classify(delta)
            if mover_is_me and tag != "ok":
                points.append({
                    "ply": ply,
                    "move": move.uci(),
                    "best": best_uci,
                    "cp_before": cp_before,
                    "cp_after": cp_after,
                    "delta": delta,
                    "tag": tag,
                })

        # 按 delta 降序取前 N
        points.sort(key=lambda x: -x["delta"])
        points = points[:max_points]
        points.sort(key=lambda x: x["ply"])

        result = {
            "depth": STOCKFISH_ANALYZE_DEPTH,
            "my_color": my_color,
            "critical_points": points,
        }
        print(f"[STOCKFISH] critical points: {len(points)}")
        return json.dumps(result, ensure_ascii=False)
    finally:
        try:
            engine.quit()
        except Exception:
            pass


def stockfish_collect_blunders(pgn_text: str, threshold_cp: int = 200):
    """收集双方所有 blunder（cp 损失 ≥ threshold_cp），含走子前后的盘面图、SAN 和走法历史（FEN 仅内部供引擎用）。
    返回 list[dict]。"""
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        print(f"[STOCKFISH] 启动失败: {e}")
        return []

    blunders = []
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return []
        board = game.board()
        limit = chess.engine.Limit(depth=STOCKFISH_ANALYZE_DEPTH)
        ply = 0
        for move in game.mainline_moves():
            ply += 1
            mover_is_white = (board.turn == chess.WHITE)
            fen_before = board.fen()
            history_before = san_history(board)
            board_before = render_board(board)
            move_san = board.san(move)
            try:
                info_b = engine.analyse(board, limit)
                cp_before = _cp_from_score(info_b["score"], pov_white=mover_is_white)
                best = info_b.get("pv", [None])[0]
                best_uci = best.uci() if best else ""
                best_san = board.san(best) if best else ""
            except Exception as e:
                print(f"[STOCKFISH] analyse before failed: {e}")
                break
            board.push(move)
            fen_after = board.fen()
            board_after = render_board(board)
            try:
                info_a = engine.analyse(board, limit)
                cp_after = _cp_from_score(info_a["score"], pov_white=mover_is_white)
            except Exception as e:
                print(f"[STOCKFISH] analyse after failed: {e}")
                break
            delta = cp_before - cp_after
            if delta >= threshold_cp:
                blunders.append({
                    "ply": ply,
                    "side": "白" if mover_is_white else "黑",
                    "move": move.uci(),
                    "san": move_san,
                    "best": best_uci,
                    "best_san": best_san,
                    "history": history_before,
                    "board_before": board_before,
                    "board_after": board_after,
                    "cp_before": cp_before,
                    "cp_after": cp_after,
                    "delta": delta,
                    "fen_before": fen_before,
                    "fen_after": fen_after,
                })
        return blunders
    finally:
        try:
            engine.quit()
        except Exception:
            pass


def stockfish_eval_move(fen_before: str, move_uci: str):
    """评估在 fen_before 局面下走 move_uci 的好坏。
    返回 dict: {cp_before, cp_after, delta, best, comment}"""
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        return {"error": f"启动失败: {e}"}
    try:
        board = chess.Board(fen_before)
        mover_is_white = (board.turn == chess.WHITE)
        limit = chess.engine.Limit(depth=STOCKFISH_ANALYZE_DEPTH)
        info_b = engine.analyse(board, limit)
        cp_before = _cp_from_score(info_b["score"], pov_white=mover_is_white)
        best = info_b.get("pv", [None])[0]
        best_uci = best.uci() if best else ""

        try:
            mv = chess.Move.from_uci(move_uci)
        except Exception:
            return {"error": f"非法 UCI: {move_uci}"}
        if mv not in board.legal_moves:
            return {"error": f"不合法走法: {move_uci}", "best": best_uci,
                    "cp_before": cp_before}
        board.push(mv)
        info_a = engine.analyse(board, limit)
        cp_after = _cp_from_score(info_a["score"], pov_white=mover_is_white)
        delta = cp_before - cp_after
        verdict = _classify(delta)
        return {
            "cp_before": cp_before,
            "cp_after": cp_after,
            "delta": delta,
            "best": best_uci,
            "verdict": verdict,
        }
    finally:
        try:
            engine.quit()
        except Exception:
            pass


def stockfish_opening_evals(uci_list: list[str], my_white: bool, max_moves: int,
                            floor_cp: int) -> list[tuple[int, int]]:
    """逐个评估我方前 max_moves 步走完后的局面（我方视角 cp），返回 [(uci_list 下标, cp)]。
    评到第一个低于 floor_cp 的就停（该项也包含在内），后面的局面没有再评的意义。启动失败返回 []。"""
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        print(f"[BOOK] Stockfish 启动失败: {e}")
        return []
    out: list[tuple[int, int]] = []
    try:
        board = chess.Board()
        limit = chess.engine.Limit(depth=STOCKFISH_ANALYZE_DEPTH)
        for i, uci in enumerate(uci_list):
            mine = (board.turn == chess.WHITE) == my_white
            board.push_uci(uci)
            if not mine:
                continue
            cp = _cp_from_score(engine.analyse(board, limit)["score"], pov_white=my_white)
            out.append((i, cp))
            if cp < floor_cp or len(out) >= max_moves:
                break
        return out
    finally:
        try:
            engine.quit()
        except Exception:
            pass
