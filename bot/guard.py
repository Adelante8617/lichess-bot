"""落子前的丢子守卫：按规则模拟走完某步后，对方在每个格上的吃子交换能净得多少分。

这是对局中唯一由程序"算"出的东西，只覆盖最朴素的一类失误：走完之后对方直接吃子就能净赚
（如象走到被兵控制的格、高价值子吃被保护的低价值子）。捉双、牵制、将杀等战术不在此列，仍由模型判断。
算法是标准的静态交换（SEE）：双方轮流用价值最低的子在同一格上吃，任何一方都可以停手；
用合法着法生成，自动处理牵制、X 光和将军。

verify_line 用来核对模型为弃子给出的变化：按规则摆一遍，只报告是否合法、终点子力，不评价局面。
"""
import re

import chess

from .boardtext import COLOR_ZH, PIECE_VALUES, display_san, material_points, parse_model_move

_ORDER_VALUE = {**PIECE_VALUES, chess.KING: 100}  # 只用于"先用哪个子吃"的排序：王最后上
MAX_LINE_PLIES = 16


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


def _best_capture(board: chess.Board) -> tuple[int, list[str], int | None]:
    """轮到走的一方对 对方 所有子（王除外）发起吃子交换，取净得最多的：(净得, 序列, 格)。"""
    best, best_line, best_sq = 0, [], None
    if board.is_game_over():
        return best, best_line, best_sq
    for sq in chess.SquareSet(board.occupied_co[not board.turn]):
        if board.piece_type_at(sq) == chess.KING:
            continue
        gain, line = _exchange(board, sq)
        if gain > best:
            best, best_line, best_sq = gain, line, sq
    return best, best_line, best_sq


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
    best, best_line, best_sq = _best_capture(after)
    return {"loss": best - gained, "line": best_line,
            "square": chess.square_name(best_sq) if best_sq is not None else ""}


def risk_text(board: chess.Board, move: chess.Move, risk: dict) -> str:
    """把模拟结果写成给模型看的事实描述。"""
    san = display_san(board, move)
    line = " ".join(risk["line"])
    return (f"走 {san} 之后，对方可以在 {risk['square']} 上吃子：按双方都用价值最低的子在该格轮流吃、"
            f"吃亏的一方停手来模拟，交换序列为 {line}，单看这一格的交换，我方净亏 {risk['loss']} 分"
            f"（兵1 马象3 车5 后9，已算上 {san} 本身吃到的子）。")


def _tokens(line: str) -> list[str]:
    """'12.Bg5 hxg5 13. Nxg5' / 'Bg5, hxg5' → ['Bg5', 'hxg5', 'Nxg5']，去掉回合编号与省略号。"""
    out = []
    for t in re.split(r"[\s,，;；→>]+", line or ""):
        t = re.sub(r"^\d+\.+", "", t.strip()).strip(".…")
        if t:
            out.append(t)
    return out


def verify_line(board: chess.Board, move: chess.Move, line: str, min_loss: int, square: str) -> dict:
    """按规则摆模型给的变化（不以 move 开头时自动补上），返回 {"ok": bool, "text": 给模型 / 日志看的事实}。
    ok 的条件：变化全部合法；对方第一步就在 square 上吃子（论证的是"对方吃了之后怎样"，
    而不是让对方不吃）；终点将杀对方，或终点子力净亏 < min_loss（终点轮到对方走时，
    再扣掉对方立即可做的最优吃子交换）。之后的对方应着由模型自己给，程序不判断它是不是最强。"""
    me = board.turn
    tokens = _tokens(line)
    b = board.copy()
    if not tokens or parse_model_move(b, tokens[0]) != move:
        tokens = [display_san(board, move)] + tokens
    played = []
    for i, t in enumerate(tokens[:MAX_LINE_PLIES], 1):
        mv = parse_model_move(b, t)
        if mv is None:
            return {"ok": False, "text": (f"变化第 {i} 步 {t} 不合法（此时轮到{COLOR_ZH[b.turn]}），"
                                          f"只能摆到 {' '.join(played) or '（无）'}。")}
        if i == 2 and square and chess.square_name(mv.to_square) != square:
            return {"ok": False, "text": (f"变化里对方第一步走了 {display_san(b, mv)}，没有在 {square} 上吃子；"
                                          f"要说明这步成立，变化需要从对方在 {square} 吃子之后讲起。")}
        played.append(display_san(b, mv))
        b.push(mv)
    shown = " ".join(played)
    if b.is_checkmate():
        return {"ok": b.turn != me, "text": f"变化 {shown} 合法，终点{COLOR_ZH[b.turn]}被将死。"}
    net = (material_points(b, me) - material_points(b, not me)) \
        - (material_points(board, me) - material_points(board, not me))
    extra = ""
    if b.turn != me:
        gain, cap, sq = _best_capture(b)
        if gain > 0:
            net -= gain
            extra = (f"（终点轮到对方，已扣除对方在 {chess.square_name(sq)} 上还能吃到的 {gain} 分："
                     f"{' '.join(cap)}）")
    return {"ok": -net < min_loss, "text": f"变化 {shown} 合法，摆到终点我方子力净得失 {net:+d} 分{extra}。"}
