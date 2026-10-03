"""把棋盘转成给模型 / 日志看的文字：盘面、子力清单、SAN、局面元信息、PGN。只陈述事实，不做判断。"""
from datetime import datetime

import chess
import chess.pgn


def render_board(board: chess.Board) -> str:
    """渲染带坐标的 ASCII 棋盘。
    白子用大写并加 'W:' 前缀的图例，黑子小写并加 'B:'，空格用 '.'。
    便于 LLM 一眼看清双方棋子分布，避免误用对方子力。"""
    rows = []
    rows.append("   a  b  c  d  e  f  g  h")
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            sq = chess.square(file, rank)
            p = board.piece_at(sq)
            if p is None:
                cells.append(" .")
            else:
                ch = p.symbol()  # 大写=白, 小写=黑
                tag = "W" if ch.isupper() else "B"
                cells.append(f"{tag}{ch.upper()}")
        rows.append(f"{rank+1}  " + " ".join(cells))
    rows.append("   a  b  c  d  e  f  g  h")
    return "\n".join(rows)


PIECE_ZH = {chess.KING: "王", chess.QUEEN: "后", chess.ROOK: "车",
            chess.BISHOP: "象", chess.KNIGHT: "马", chess.PAWN: "兵"}
COLOR_ZH = {chess.WHITE: "白方", chess.BLACK: "黑方"}


def piece_lists(board: chess.Board) -> tuple[str, str]:
    """返回 (白方子力描述, 黑方子力描述)，按子种分组并标格子，如 '王 e1；后 d1；车 a1, h1'。"""
    def list_for(color: bool) -> str:
        parts = []
        for pt in (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
            sqs = sorted(chess.square_name(sq) for sq in board.pieces(pt, color))
            if sqs:
                parts.append(f"{PIECE_ZH[pt]} {', '.join(sqs)}")
        return "；".join(parts)
    return list_for(chess.WHITE), list_for(chess.BLACK)


def strip_check(san: str) -> str:
    """去掉 SAN 的 +/# 后缀，用于比较 move 与 pv[0]。"""
    return san.rstrip("+#")


def display_san(board: chess.Board, move: chess.Move) -> str:
    """列表里展示用的 SAN：保留将军符号 +；将死的 # 也显示成 +，
    这样只告诉模型"这步会将军"，不额外暴露"这步直接将死"。"""
    san = board.san(move)
    return san[:-1] + "+" if san.endswith("#") else san


def legal_san_map(board: chess.Board) -> dict[str, chess.Move]:
    """{展示用 SAN: Move}，同一局面下 SAN 唯一。"""
    return {display_san(board, m): m for m in board.legal_moves}


def parse_model_move(board: chess.Board, text: str) -> chess.Move | None:
    """把模型给的着法解析成合法 Move。主要接受 SAN（容忍 +/#/!/? 后缀、0-0 写法），
    顺带兼容 UCI。解析失败或非法返回 None。"""
    text = (text or "").strip().rstrip("+#!?")
    if not text:
        return None
    for cand in (text, text.replace("0", "O")):
        try:
            return board.parse_san(cand)
        except ValueError:
            pass
    try:
        mv = chess.Move.from_uci(text)
        return mv if mv in board.legal_moves else None
    except ValueError:
        return None


def describe_last_move(prev_board: chess.Board, uci: str) -> str:
    """把对方上一步描述成人话：谁、什么子、从哪到哪、SAN、是否吃子/升变/易位。"""
    mv = chess.Move.from_uci(uci)
    piece = prev_board.piece_at(mv.from_square)
    who = COLOR_ZH[prev_board.turn]
    san = prev_board.san(mv)
    parts = [f"{who}走了 {san}"]
    if piece:
        parts.append(f"（{PIECE_ZH[piece.piece_type]} "
                     f"{chess.square_name(mv.from_square)}→{chess.square_name(mv.to_square)}）")
    if prev_board.is_en_passant(mv):
        parts.append("，吃过路兵")
    elif prev_board.is_capture(mv):
        victim = prev_board.piece_at(mv.to_square)
        parts.append(f"，吃掉了{COLOR_ZH[not prev_board.turn]}的{PIECE_ZH[victim.piece_type]}")
    if mv.promotion:
        parts.append(f"，升变为{PIECE_ZH[mv.promotion]}")
    if prev_board.is_castling(mv):
        parts.append("，王车易位")
    return "".join(parts)


def board_meta(board: chess.Board) -> str:
    """当前局面的非棋子信息：轮到谁、是否被将军、易位权、吃过路兵、50 步计数。"""
    lines = [f"轮到: {COLOR_ZH[board.turn]}"]
    lines.append("你正被将军，必须应将" if board.is_check() else "当前未被将军")
    for color in (chess.WHITE, chess.BLACK):
        rights = []
        if board.has_kingside_castling_rights(color):
            rights.append("短易位 O-O")
        if board.has_queenside_castling_rights(color):
            rights.append("长易位 O-O-O")
        lines.append(f"{COLOR_ZH[color]}尚保留的易位权: " + ("、".join(rights) if rights else "无"))
    lines.append("（易位权仅表示王和对应车没动过，此刻能否易位以合法着法列表为准）")
    if board.ep_square is not None and board.has_legal_en_passant():
        lines.append(f"可吃过路兵，目标格 {chess.square_name(board.ep_square)}")
    else:
        lines.append("当前无法吃过路兵")
    lines.append(f"距上次吃子/动兵已过 {board.halfmove_clock} 个半回合（满 100 判和）")
    return "\n".join(lines)


def material_text(board: chess.Board, color: bool) -> str:
    return "".join(f"{PIECE_ZH[pt]}{len(board.pieces(pt, color))}"
                   for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)
                   if board.pieces(pt, color)) or "仅剩王"


def game_phase(board: chess.Board) -> str:
    if board.fullmove_number <= 10:
        return "开局"
    pieces = sum(len(board.pieces(pt, c)) * v
                 for pt, v in ((chess.QUEEN, 9), (chess.ROOK, 5), (chess.BISHOP, 3), (chess.KNIGHT, 3))
                 for c in (chess.WHITE, chess.BLACK))
    return "残局" if pieces <= 26 else "中局"


def material_lead(board: chess.Board) -> int:
    """轮到走棋一方的子力分差（兵1 马象3 车5 后9），正数表示我方领先。"""
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    return sum(v * (len(board.pieces(pt, board.turn)) - len(board.pieces(pt, not board.turn)))
               for pt, v in values.items())


def uci_to_san(fen: str, uci: str) -> str:
    """在 fen 局面下把 UCI 着法转成 SAN；失败则原样返回 uci。"""
    try:
        b = chess.Board(fen)
        return b.san(chess.Move.from_uci(uci))
    except Exception:
        return uci


def san_history(board: chess.Board) -> str:
    """把 board 上已走的着法转成带回合号的 SAN 序列，如 '1. e4 e5 2. Nf3 Nc6'。"""
    tmp = board.root()
    parts = []
    for mv in board.move_stack:
        if tmp.turn == chess.WHITE:
            parts.append(f"{tmp.fullmove_number}.")
        parts.append(tmp.san(mv))
        tmp.push(mv)
    return " ".join(parts)


def build_pgn(moves_uci: list, result: str) -> str:
    game = chess.pgn.Game()
    game.headers["Event"] = "Lichess LLM Bot"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Result"] = result
    node = game
    b = chess.Board()
    for uci in moves_uci:
        try:
            mv = chess.Move.from_uci(uci)
            if mv not in b.legal_moves:
                break
            node = node.add_variation(mv)
            b.push(mv)
        except Exception:
            break
    return str(game)
