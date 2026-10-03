"""
本地对局：不连 Lichess，用 python-chess 维护棋盘，让 LLM 与对手对弈。
复用 main.py 的 get_llm_move（决策）和赛后复盘，行为与线上一致。

用法：
  python local_play.py                          # 默认：LLM(随机执色) vs 随机走子（最弱对手）
  python local_play.py --color white
  # Stockfish 对手，由弱到强（参数可组合）：
  python local_play.py --opponent stockfish --depth 1 --blunder-rate 0.6   # 60% 随机 + 只搜 1 层
  python local_play.py --opponent stockfish --depth 1 --blunder-rate 0.3
  python local_play.py --opponent stockfish --elo 1320                     # UCI_Elo 下限 1320
  python local_play.py --opponent stockfish --skill 0
  python local_play.py --opponent stockfish --depth 1
  python local_play.py --opponent stockfish --skill 5 --movetime 0.1       # 之前三局用的对手，强很多
  python local_play.py --opponent human         # 你在终端输入 SAN/UCI 走法
  python local_play.py --opponent llm           # LLM 自己和自己下
  python local_play.py --no-review              # 不做赛后复盘（不写经验库）
"""
import argparse
import os
import random
from datetime import datetime

import chess
import chess.engine
import chess.pgn

import main as bot  # 导入不会启动 Lichess 主循环
from bot.live import live


def parse_human_move(board: chess.Board, text: str) -> chess.Move | None:
    text = text.strip()
    try:
        return board.parse_uci(text)
    except ValueError:
        pass
    try:
        return board.parse_san(text)
    except ValueError:
        return None


def opponent_move(board, kind, engine, limit, llm_ctx, blunder_rate=0.0):
    if kind == "random":
        return random.choice(list(board.legal_moves))
    if kind == "stockfish":
        if blunder_rate and random.random() < blunder_rate:
            print("[OPP-WEAK] 随机走子")
            return random.choice(list(board.legal_moves))
        return engine.play(board, limit).move
    if kind == "human":
        print(board)
        while True:
            mv = parse_human_move(board, input(f"你的走法({'白' if board.turn else '黑'}): "))
            if mv and mv in board.legal_moves:
                return mv
            print("非法走法，请重输（SAN 如 Nf3，或 UCI 如 g1f3）")
    # kind == "llm"：对手也是同一套 LLM
    prev, last = llm_ctx
    uci, *_ = bot.get_llm_move(board, board.ply() + 1, prev, last)
    return chess.Move.from_uci(uci)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", choices=["white", "black", "random"], default="random")
    ap.add_argument("--opponent", choices=["random", "stockfish", "human", "llm"], default="random",
                    help="random=每步随机合法着法（最弱，默认）")
    ap.add_argument("--skill", type=int, default=5, help="Stockfish Skill Level 0-20")
    ap.add_argument("--movetime", type=float, default=0.1, help="Stockfish 每步秒数")
    ap.add_argument("--elo", type=int, default=None,
                    help="启用 UCI_LimitStrength 并设置 Elo（Stockfish 18 范围 1320-3190，会覆盖 --skill）")
    ap.add_argument("--depth", type=int, default=None, help="限制搜索深度（如 1、2），设置后忽略 --movetime")
    ap.add_argument("--nodes", type=int, default=None, help="限制搜索节点数（如 50），设置后忽略 --movetime")
    ap.add_argument("--blunder-rate", type=float, default=0.0,
                    help="每步以该概率改走随机合法着法，0-1，用来把对手调到想要的强度")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--no-review", action="store_true")
    args = ap.parse_args()

    my_white = {"white": True, "black": False}.get(args.color, random.random() < 0.5)
    my_color = "白" if my_white else "黑"
    print(f"=== 本地对局：我执{my_color}，对手={args.opponent} ===")
    limit_kw = ({"depth": args.depth} if args.depth else {"nodes": args.nodes} if args.nodes
                else {"time": args.movetime})
    limit = chess.engine.Limit(**limit_kw)
    sf_desc = f"Elo {args.elo}" if args.elo else f"Skill {args.skill}"
    if args.depth:
        sf_desc += f", depth {args.depth}"
    elif args.nodes:
        sf_desc += f", nodes {args.nodes}"
    if args.blunder_rate:
        sf_desc += f", 随机 {args.blunder_rate:.0%}"
    opp_label = {"random": "随机走子", "stockfish": f"Stockfish ({sf_desc})",
                 "human": "人类", "llm": "LLM 自对弈"}[args.opponent]
    live.start_game(datetime.now().strftime("local_%H%M%S"), "local", my_color, opp_label)

    engine = None
    if args.opponent == "stockfish":
        engine = chess.engine.SimpleEngine.popen_uci(bot.STOCKFISH_PATH)
        if args.elo:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo})
        else:
            engine.configure({"Skill Level": max(0, min(20, args.skill))})

    board = chess.Board()
    uci_list: list[str] = []
    move_log = []  # (ply, fen, move, think)，供 post_game_review 使用
    snapshots: list = []  # 待验证快照，赛后由 Stockfish 验证后入库

    try:
        while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
            prev = chess.Board()
            for u in uci_list[:-1]:
                prev.push_uci(u)
            last = uci_list[-1] if uci_list else None

            if board.turn == my_white:
                print(f"\n--- ply {board.ply() + 1} 我方走子 ---")
                print(board)
                move, think, opp_intent, obs = bot.get_llm_move(
                    board, board.ply() + 1, prev if last else None, last)
                print(f"[THINK] {think}\n[MOVE]  {move}")
                move_log.append((board.ply() + 1, board.fen(), move, think))
                bot.record_snapshot(snapshots, board, move, obs)
                mv = chess.Move.from_uci(move)
            else:
                mv = opponent_move(board, args.opponent, engine, limit,
                                   (prev if last else None, last), args.blunder_rate)
                print(f"[OPP]   {board.san(mv)}")

            uci_list.append(mv.uci())
            board.push(mv)
            live.sync(uci_list)
    finally:
        if engine:
            engine.quit()

    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
    print(f"\n=== 对局结束：{result}  ({board.ply()} plies) ===")

    live.set_status("reviewing" if not args.no_review else "finished", result)
    pgn_text = bot.build_pgn(uci_list, result)
    os.makedirs("games", exist_ok=True)
    path = os.path.join("games", f"{datetime.now():%Y%m%d_%H%M%S}_local.pgn")
    with open(path, "w", encoding="utf-8") as f:
        f.write(pgn_text)
    print(f"PGN saved to {path}")

    if not args.no_review:
        bot.post_game_review(pgn_text, result, my_color, move_log)
        try:
            bot.blunder_deep_review(pgn_text, result, my_color)
        except Exception as e:
            print(f"[BLUNDER-REVIEW] failed: {e}")
        bot.commit_verified_snapshots(snapshots, result, my_color)
    live.set_status("finished", result)


if __name__ == "__main__":
    main()
