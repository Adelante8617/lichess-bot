"""
棋盘的"原始感知"：只按规则列出客观关系，不做任何判断或评估。

- relations_text：每个子控制哪些格、攻击/保护了谁、被谁攻击/保护；
  每条直线上依次碰到的子（用于看出 X 光、牵制、闪击，但不替模型下"这是牵制"的结论）；
  王周边格被对方控制的情况；兵的分布与开放线。
- play_line：分析棋盘。按顺序摆一串着法，报告每步是否合法、摆完后的局面与关系，
  不评估局面、不给分数、不推荐着法。

悬子、捉双、牵制、谁好谁坏、下一步该走什么——全部留给模型自己判断。
"""
import chess

from .boardtext import PIECE_ZH, parse_model_move, render_board

SIDE_ZH = {chess.WHITE: "白", chess.BLACK: "黑"}
ORDER = (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)
DIRS = {chess.ROOK: ((0, 1), (0, -1), (1, 0), (-1, 0)),
        chess.BISHOP: ((1, 1), (1, -1), (-1, 1), (-1, -1))}
DIRS[chess.QUEEN] = DIRS[chess.ROOK] + DIRS[chess.BISHOP]
MAX_LINE_PLIES = 12


def pname(board: chess.Board, sq: int) -> str:
    """'白马f3' 形式。"""
    p = board.piece_at(sq)
    return f"{SIDE_ZH[p.color]}{PIECE_ZH[p.piece_type]}{chess.square_name(sq)}"


def _names(board: chess.Board, squares) -> str:
    return "、".join(pname(board, s) for s in sorted(squares)) or "无"


def _piece_line(board: chess.Board, sq: int) -> str:
    p = board.piece_at(sq)
    ctrl = board.attacks(sq)
    empty = sorted(chess.square_name(s) for s in ctrl if board.piece_at(s) is None)
    hits_enemy = [s for s in ctrl if board.piece_at(s) and board.piece_at(s).color != p.color]
    hits_own = [s for s in ctrl if board.piece_at(s) and board.piece_at(s).color == p.color]
    parts = [f"{pname(board, sq)}：控制空格 {' '.join(empty) or '无'}"]
    if hits_enemy:
        parts.append(f"攻击 {_names(board, hits_enemy)}")
    if hits_own:
        parts.append(f"保护 {_names(board, hits_own)}")
    parts.append(f"被对方攻击 {_names(board, board.attackers(not p.color, sq))}")
    parts.append(f"被己方保护 {_names(board, board.attackers(p.color, sq))}")
    return "；".join(parts)


def _lines(board: chess.Board) -> list[str]:
    """长程子沿每个方向依次碰到的前两个子，只列第二个子属于对方的情况
    （这正是 X 光 / 牵制 / 串击 / 闪击会出现的几何形状）。"""
    out = []
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP):
            for sq in sorted(board.pieces(pt, color)):
                for df, dr in DIRS[pt]:
                    f, r = chess.square_file(sq), chess.square_rank(sq)
                    hit = []
                    while len(hit) < 2:
                        f, r = f + df, r + dr
                        if not (0 <= f < 8 and 0 <= r < 8):
                            break
                        s = chess.square(f, r)
                        if board.piece_at(s):
                            hit.append(s)
                    if len(hit) == 2 and board.piece_at(hit[1]).color != color:
                        out.append(f"{pname(board, sq)} 所在线上：先是 {pname(board, hit[0])}，"
                                   f"其后是 {pname(board, hit[1])}")
    return out


def _king_zone(board: chess.Board, color: bool) -> str:
    k = board.king(color)
    if k is None:
        return ""
    zone = [s for s in chess.SquareSet(chess.BB_KING_ATTACKS[k]) | chess.SquareSet.from_square(k)]
    hit = []
    for s in sorted(zone):
        att = board.attackers(not color, s)
        if att:
            hit.append(f"{chess.square_name(s)}←{_names(board, att)}")
    return (f"{SIDE_ZH[color]}王{chess.square_name(k)} 周围被对方控制的格："
            + ("；".join(hit) if hit else "无"))


def _pawns(board: chess.Board) -> list[str]:
    out = []
    files = "abcdefgh"
    for color in (chess.WHITE, chess.BLACK):
        by_file = {}
        for s in board.pieces(chess.PAWN, color):
            by_file.setdefault(chess.square_file(s), []).append(chess.square_name(s))
        desc = " ".join(",".join(sorted(v)) for _, v in sorted(by_file.items())) or "无"
        passed = []
        for s in board.pieces(chess.PAWN, color):
            f, r = chess.square_file(s), chess.square_rank(s)
            ahead = range(r + 1, 8) if color == chess.WHITE else range(0, r)
            blockers = [t for t in board.pieces(chess.PAWN, not color)
                        if abs(chess.square_file(t) - f) <= 1 and chess.square_rank(t) in ahead]
            if not blockers:
                passed.append(chess.square_name(s))
        line = f"{SIDE_ZH[color]}兵：{desc}"
        if passed:
            line += f"（前方及相邻线上没有对方兵：{' '.join(sorted(passed))}）"
        out.append(line)
    w = {chess.square_file(s) for s in board.pieces(chess.PAWN, chess.WHITE)}
    b = {chess.square_file(s) for s in board.pieces(chess.PAWN, chess.BLACK)}
    out.append("无兵的线：" + (" ".join(files[f] for f in range(8) if f not in w | b) or "无"))
    out.append("只有黑兵的线：" + (" ".join(files[f] for f in sorted(b - w)) or "无")
               + "；只有白兵的线：" + (" ".join(files[f] for f in sorted(w - b)) or "无"))
    return out


def relations_text(board: chess.Board) -> str:
    me = board.turn
    sections = []
    for color, title in ((me, "我方"), (not me, "对方")):
        rows = [_piece_line(board, sq) for pt in ORDER for sq in sorted(board.pieces(pt, color))]
        sections.append(f"[{title}（{SIDE_ZH[color]}）各子]\n" + "\n".join(rows))
    lines = _lines(board)
    sections.append("[直线上的前后两子]\n" + ("\n".join(lines) if lines else "无"))
    sections.append("[王的周边]\n" + _king_zone(board, me) + "\n" + _king_zone(board, not me))
    sections.append("[兵形]\n" + "\n".join(_pawns(board)))
    return "\n\n".join(sections)


def _material(board: chess.Board, color: bool) -> str:
    return " ".join(f"{PIECE_ZH[pt]}{len(board.pieces(pt, color))}" for pt in ORDER[1:]
                    if board.pieces(pt, color)) or "仅剩王"


def play_line(board: chess.Board, moves: list) -> str:
    """在 board 的副本上依次摆 moves，返回过程与终点局面的原始描述。"""
    b = board.copy()
    steps = []
    moves = [str(m) for m in (moves or [])][:MAX_LINE_PLIES]
    stopped = ""
    for i, text in enumerate(moves, 1):
        mv = parse_model_move(b, text)
        if mv is None:
            stopped = (f"第 {i} 步 '{text}' 在该局面不合法（此时轮到{SIDE_ZH[b.turn]}方），"
                       f"已停在第 {i - 1} 步之后。")
            break
        note = []
        if b.is_capture(mv):
            victim = (chess.PAWN if b.is_en_passant(mv) else b.piece_at(mv.to_square).piece_type)
            note.append(f"吃{SIDE_ZH[not b.turn]}{PIECE_ZH[victim]}")
        san = b.san(mv)
        who = SIDE_ZH[b.turn]
        b.push(mv)
        if b.is_check():
            note.append("将军")
        steps.append(f"{i}. {who} {san.rstrip('#').rstrip('+')}" + (f"（{'，'.join(note)}）" if note else ""))

    out = ["[摆出的着法]", "\n".join(steps) or "（无）"]
    if stopped:
        out.append(stopped)
    if b.is_checkmate():
        out.append(f"终点：{SIDE_ZH[b.turn]}方被将死。")
    elif b.is_stalemate():
        out.append(f"终点：{SIDE_ZH[b.turn]}方无子可动，逼和。")
    elif b.is_insufficient_material():
        out.append("终点：双方子力不足以将杀，和棋。")
    elif b.can_claim_draw():
        out.append("终点：可按重复局面或 50 步规则判和。")
    out.append(f"终点轮到{SIDE_ZH[b.turn]}方走" + ("，正被将军" if b.is_check() else ""))
    out.append(f"子力：白 {_material(b, chess.WHITE)}；黑 {_material(b, chess.BLACK)}")
    out.append(render_board(b))
    legal = [b.san(m) for m in b.legal_moves]
    legal = [s[:-1] + "+" if s.endswith("#") else s for s in legal]
    out.append(f"{SIDE_ZH[b.turn]}方合法着法：{', '.join(legal) or '无'}")
    out.append(relations_text(b))
    return "\n".join(out)
