"""
用 Laya（convaiinnovations/laya，非自回归的 System 1 决策模型）下棋的本地对局。
与 LLM 下棋完全独立：不导入 main.py，不需要 LLM / Embedding 的 key，不读写经验库，不做 LLM 复盘。

Laya 只会做"从给定选项里选一个"的分类，本身不懂棋，且一个 choice 问题的全部选项
共享约 192 token 的预算，塞不下整盘合法着法。所以分两步：
  1. 生成候选着法：
       stockfish → Stockfish MultiPV 前 K 步（默认，候选质量有保证，Laya 在其中挑）
       legal     → 全部合法着法，按每组 K 个分组淘汰，逐轮选出胜者（纯 Laya，无引擎辅助）
  2. 把局面（FEN + 棋盘 + 子力 + 上一步）作为 state，候选着法作为 choice 选项，
     Laya 一次前向给出每个候选的概率，取最高者。
候选顺序默认打乱，避免 Stockfish 排序被 Laya 的位置偏好"蹭"到。

用法：
  python laya_play.py                                      # Laya(随机执色) vs 随机走子
  python laya_play.py --opponent stockfish --depth 1 --blunder-rate 0.5
  python laya_play.py --candidates legal                   # 不借助引擎，纯 Laya 选着
  python laya_play.py --k 8 --cand-depth 10                # 候选数 / 候选搜索深度
  python laya_play.py --opponent human --color white
  python laya_play.py --opponent llm                       # 对手用原 LLM（需 .env 里的 LLM 配置）

首次运行会从 HuggingFace 下载约 1.7GB 的权重。
"""
import os

# torch 与 numpy 各带一份 OpenMP 运行时，Windows 上不设此变量 import torch 会直接崩
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import random
import time
from datetime import datetime

import chess
import chess.engine
import chess.pgn
from dotenv import load_dotenv

from live import live

load_dotenv()

STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "stockfish")
LAYA_MODEL = os.getenv("LAYA_MODEL", "convaiinnovations/laya")
# 当前环境没 pip 装 laya 时，从这个源码目录导入（如带 CUDA torch 的其他环境 + E:\laya 源码）
LAYA_SRC = os.getenv("LAYA_SRC", r"E:\laya")


def import_laya():
    try:
        import laya
    except ModuleNotFoundError:
        if not os.path.isdir(os.path.join(LAYA_SRC, "laya")):
            raise ModuleNotFoundError(
                f"当前环境未安装 laya，且 LAYA_SRC={LAYA_SRC} 下没有 laya 源码；"
                f"请 pip install laya 或在 .env 中设置 LAYA_SRC") from None
        import sys
        sys.path.insert(0, LAYA_SRC)
        import laya
        print(f"[LAYA] 使用源码 {LAYA_SRC}")
    return laya

PIECE_EN = {chess.KING: "king", chess.QUEEN: "queen", chess.ROOK: "rook",
            chess.BISHOP: "bishop", chess.KNIGHT: "knight", chess.PAWN: "pawn"}
PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
COLOR_EN = {chess.WHITE: "White", chess.BLACK: "Black"}


# ---------- 局面 / 着法的文字描述（Laya 的英文 checkpoint 读英文效果最好） ----------

def _piece_list(board: chess.Board, color: bool) -> str:
    parts = []
    for pt in (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
        sqs = sorted(chess.square_name(s) for s in board.pieces(pt, color))
        if sqs:
            parts.append(f"{PIECE_EN[pt]} {' '.join(sqs)}")
    return "; ".join(parts)


def _material(board: chess.Board, color: bool) -> int:
    return sum(len(board.pieces(pt, color)) * v for pt, v in PIECE_VALUE.items())


def describe_position(board: chess.Board, last_move: str | None) -> str:
    me = board.turn
    lines = [
        f"Chess position. {COLOR_EN[me]} to move.",
        f"FEN: {board.fen()}",
        f"White pieces: {_piece_list(board, chess.WHITE)}",
        f"Black pieces: {_piece_list(board, chess.BLACK)}",
        f"Material: White {_material(board, chess.WHITE)}, Black {_material(board, chess.BLACK)}",
    ]
    if last_move:
        lines.append(f"Opponent's last move: {last_move}")
    if board.is_check():
        lines.append(f"{COLOR_EN[me]} is in check and must respond.")
    lines.append("Board (uppercase = White, lowercase = Black):")
    lines.append(str(board))
    return "\n".join(lines)


def describe_move(board: chess.Board, mv: chess.Move) -> str:
    piece = board.piece_at(mv.from_square)
    desc = [f"{PIECE_EN[piece.piece_type]} {chess.square_name(mv.from_square)}"
            f" to {chess.square_name(mv.to_square)}"]
    if board.is_castling(mv):
        desc = ["castle kingside" if chess.square_file(mv.to_square) > 4 else "castle queenside"]
    if board.is_en_passant(mv):
        desc.append("captures pawn en passant")
    elif board.is_capture(mv):
        desc.append(f"captures {PIECE_EN[board.piece_at(mv.to_square).piece_type]}")
    if mv.promotion:
        desc.append(f"promotes to {PIECE_EN[mv.promotion]}")
    if board.gives_check(mv):
        desc.append("gives check")
    return ", ".join(desc)


# ---------- Laya 选着 ----------

class LayaPlayer:
    def __init__(self, model_id: str = LAYA_MODEL, candidates: str = "stockfish", k: int = 6,
                 cand_depth: int = 8, head_max_len: int = 256, shuffle: bool = True,
                 device: str | None = None):
        laya = import_laya()  # 延迟导入：torch 很重，只在真正用 Laya 时加载
        print(f"[LAYA] loading {model_id} ...")
        t = time.time()
        self.agent = laya.load(model_id, device=device)
        print(f"[LAYA] loaded in {time.time() - t:.1f}s ({self.agent})")
        self.candidates = candidates
        self.k = max(2, k)
        self.cand_depth = cand_depth
        self.head_max_len = head_max_len
        self.shuffle = shuffle
        self.engine = (chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
                       if candidates == "stockfish" else None)

    def close(self):
        if self.engine:
            self.engine.quit()
            self.engine = None

    def _stockfish_candidates(self, board: chess.Board) -> list[chess.Move]:
        infos = self.engine.analyse(board, chess.engine.Limit(depth=self.cand_depth),
                                    multipv=self.k)
        return [i["pv"][0] for i in infos if i.get("pv")]

    def _ask(self, board: chess.Board, state: str, moves: list[chess.Move]) -> dict[str, float]:
        """让 Laya 在 moves 中选一个，返回 {SAN: 概率}。"""
        criteria = {board.san(m): describe_move(board, m) for m in moves}
        q = {"move": {
            "type": "choice",
            "instructions": f"You are a strong chess player playing {COLOR_EN[board.turn]}. "
                            f"Which move is the best move in this position?",
            "criteria": criteria,
        }}
        r = self.agent.system_one(state, q, head_max_len=self.head_max_len)
        return r["answers"]["move"]["probabilities"]

    def choose(self, board: chess.Board, last_move: str | None = None) -> tuple[chess.Move, dict]:
        legal = list(board.legal_moves)
        if len(legal) == 1:
            return legal[0], {"note": "唯一合法着法"}
        state = describe_position(board, last_move)

        if self.candidates == "stockfish":
            cands = self._stockfish_candidates(board)
            sf_rank = {board.san(m): i + 1 for i, m in enumerate(cands)}
            if self.shuffle:
                random.shuffle(cands)
            probs = self._ask(board, state, cands)
            best = max(probs, key=probs.get)
            return board.parse_san(best), {"probabilities": probs, "sf_rank": sf_rank.get(best),
                                           "rounds": 1}

        # legal：分组淘汰，每组 k 个，胜者进入下一轮，直到剩一组
        pool = legal[:]
        if self.shuffle:
            random.shuffle(pool)
        rounds = 0
        while True:
            rounds += 1
            groups = [pool[i:i + self.k] for i in range(0, len(pool), self.k)]
            winners, probs = [], {}
            for g in groups:
                if len(g) == 1:
                    winners.append(g[0])
                    continue
                probs = self._ask(board, state, g)
                winners.append(board.parse_san(max(probs, key=probs.get)))
            if len(groups) == 1:
                return winners[0], {"probabilities": probs, "rounds": rounds}
            pool = winners


# ---------- 对手 ----------

def parse_human_move(board: chess.Board, text: str) -> chess.Move | None:
    text = text.strip()
    for parse in (board.parse_uci, board.parse_san):
        try:
            return parse(text)
        except ValueError:
            pass
    return None


def opponent_move(board, kind, engine, limit, blunder_rate, uci_list):
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
    # kind == "llm"：原 LLM 作为对手，仅在此时才导入 main
    import main as bot
    prev = chess.Board()
    for u in uci_list[:-1]:
        prev.push_uci(u)
    last = uci_list[-1] if uci_list else None
    uci, *_ = bot.get_llm_move(board, board.ply() + 1, prev if last else None, last)
    return chess.Move.from_uci(uci)


def build_pgn(uci_list: list[str], result: str, white: str, black: str) -> str:
    game = chess.pgn.Game()
    game.headers.update(Event="Local Laya game", Date=datetime.now().strftime("%Y.%m.%d"),
                        White=white, Black=black, Result=result)
    node = game
    for u in uci_list:
        node = node.add_variation(chess.Move.from_uci(u))
    return str(game)


def main():
    ap = argparse.ArgumentParser(description="Laya 决策模型下棋（独立于 LLM 下棋）")
    ap.add_argument("--color", choices=["white", "black", "random"], default="random")
    ap.add_argument("--opponent", choices=["random", "stockfish", "human", "llm"], default="random")
    ap.add_argument("--candidates", choices=["stockfish", "legal"], default="stockfish",
                    help="stockfish=引擎 MultiPV 前 K 步作候选；legal=全部合法着法分组淘汰")
    ap.add_argument("--k", type=int, default=6, help="每次交给 Laya 的候选数（选项越多每个分到的 token 越少）")
    ap.add_argument("--cand-depth", type=int, default=8, help="候选生成的 Stockfish 深度")
    ap.add_argument("--no-shuffle", action="store_true", help="不打乱候选顺序")
    ap.add_argument("--model", default=LAYA_MODEL, help="Laya checkpoint，默认 convaiinnovations/laya")
    ap.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    ap.add_argument("--seed", type=int, default=None)
    # 对手 Stockfish 强度（与 local_play.py 一致）
    ap.add_argument("--skill", type=int, default=5)
    ap.add_argument("--movetime", type=float, default=0.1)
    ap.add_argument("--elo", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--nodes", type=int, default=None)
    ap.add_argument("--blunder-rate", type=float, default=0.0)
    ap.add_argument("--max-plies", type=int, default=300)
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    my_white = {"white": True, "black": False}.get(args.color, random.random() < 0.5)
    my_color = "白" if my_white else "黑"
    print(f"=== Laya 对局：Laya 执{my_color}，对手={args.opponent}，候选={args.candidates} ===")

    player = LayaPlayer(args.model, args.candidates, args.k, args.cand_depth,
                        shuffle=not args.no_shuffle, device=args.device)

    engine = None
    limit = chess.engine.Limit(**({"depth": args.depth} if args.depth else
                                  {"nodes": args.nodes} if args.nodes else {"time": args.movetime}))
    if args.opponent == "stockfish":
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo} if args.elo
                         else {"Skill Level": max(0, min(20, args.skill))})
    opp_label = {"random": "随机走子", "stockfish": "Stockfish", "human": "人类",
                 "llm": "LLM"}[args.opponent]
    live.start_game(datetime.now().strftime("laya_%H%M%S"), "local", my_color, opp_label)

    board = chess.Board()
    uci_list: list[str] = []
    last_san = None  # 对方上一步的 SAN，写进 Laya 的局面描述
    try:
        while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
            if board.turn == my_white:
                ply = board.ply() + 1
                print(f"\n--- ply {ply} Laya 走子 ---")
                print(board)
                live.thinking(ply)
                t = time.time()
                mv, info = player.choose(board, last_san)
                probs = info.get("probabilities") or {}
                think = "Laya 候选概率：" + ", ".join(
                    f"{s} {p:.2f}" for s, p in sorted(probs.items(), key=lambda x: -x[1]))
                if info.get("sf_rank"):
                    think += f"（所选着法为 Stockfish 第 {info['sf_rank']} 推荐）"
                if info.get("note"):
                    think = info["note"]
                san = board.san(mv)
                print(f"[LAYA]  {think}  ({time.time() - t:.2f}s)\n[MOVE]  {san}")
                live.decision(ply, move_san=san, move_uci=mv.uci(), think=think,
                              opp_intent="", obs={}, warnings=[], attempts=info.get("rounds", 1),
                              fallback=False)
            else:
                mv = opponent_move(board, args.opponent, engine, limit, args.blunder_rate, uci_list)
                last_san = board.san(mv)
                print(f"[OPP]   {last_san}")
            uci_list.append(mv.uci())
            board.push(mv)
            live.sync(uci_list)
    finally:
        player.close()
        if engine:
            engine.quit()

    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
    print(f"\n=== 对局结束：{result}  ({board.ply()} plies) ===")
    live.set_status("finished", result)
    names = ("Laya", opp_label) if my_white else (opp_label, "Laya")
    os.makedirs("games", exist_ok=True)
    path = os.path.join("games", f"{datetime.now():%Y%m%d_%H%M%S}_laya.pgn")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_pgn(uci_list, result, *names))
    print(f"PGN saved to {path}")


if __name__ == "__main__":
    main()
