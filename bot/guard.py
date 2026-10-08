"""落子前的丢子守卫：按规则模拟走完某步后，对方在每个格上的吃子交换能净得多少分。

这是对局中唯一由程序"算"出的东西，只覆盖最朴素的一类失误：走完之后对方直接吃子就能净赚
（如象走到被兵控制的格、高价值子吃被保护的低价值子）。捉双、牵制、将杀等战术不在此列，仍由模型判断。
算法是标准的静态交换（SEE）：双方轮流用价值最低的子在同一格上吃，任何一方都可以停手；
用合法着法生成，自动处理牵制、X 光和将军。
"""
import chess

from .boardtext import PIECE_VALUES, display_san

_ORDER_VALUE = {**PIECE_VALUES, chess.KING: 100}  # 只用于"先用哪个子吃"的排序：王最后上


def _value(piece: chess.Piece | None) -> int:
    return PIECE_VALUES.get(piece.piece_type, 0) if piece else 0


def _exchange(board: chess.Board, square: int) -> tuple[int, list[str]]:
    """轮到走的一方在 square 上发起吃子交换，最多净得多少分（不吃为 0），以及对应的交换序列 SAN。"""
    captures = [m for m in board.legal_moves if m.to_square == square] if board.piece_at(square) else []
    if not captures:
        return 0, []
    move = min(captures, key=lambda m: _ORDER_VALUE[board.piece_type_at(m.from_square)])
    gain = _value(board.piece_at(square))
    if move.promotion:
        gain += PIECE_VALUES[move.promotion] - 1
    san = display_san(board, move)
    after = board.copy(stack=False)
    after.push(move)
    reply, line = _exchange(after, square)
    if gain - reply <= 0:
        return 0, []
    return gain - reply, [san] + line


def material_risk(board: chess.Board, move: chess.Move) -> dict:
    """走 move 之后，对方用一次吃子交换能让我方净亏多少（已扣除 move 本身吃到的子）。
    返回 {"loss": 净亏分数（≤0 表示不亏）, "line": 对方的最优交换序列, "square": 交换发生的格}。"""
    gained = _value(board.piece_at(move.to_square))
    if board.is_en_passant(move):
        gained = 1
    if move.promotion:
        gained += PIECE_VALUES[move.promotion] - 1
    after = board.copy(stack=False)
    after.push(move)
    best, best_line, best_sq = 0, [], None
    if not after.is_game_over():
        for sq in chess.SquareSet(after.occupied_co[board.turn]):
            if after.piece_type_at(sq) == chess.KING:
                continue
            gain, line = _exchange(after, sq)
            if gain > best:
                best, best_line, best_sq = gain, line, sq
    return {"loss": best - gained, "line": best_line,
            "square": chess.square_name(best_sq) if best_sq is not None else ""}


def risk_text(board: chess.Board, move: chess.Move, risk: dict) -> str:
    """把模拟结果写成给模型看的事实描述。"""
    san = display_san(board, move)
    line = " ".join(risk["line"])
    return (f"走 {san} 之后，对方可以在 {risk['square']} 上吃子：按双方都用价值最低的子吃、谁吃亏谁停手来模拟，"
            f"交换序列为 {line}，结果我方净亏 {risk['loss']} 分（兵1 马象3 车5 后9，已算上 {san} 本身吃到的子）。")
