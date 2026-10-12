"""落子前的丢子守卫：按规则模拟走完某步后，对方在每个格上的吃子交换能净得多少分。

这是对局中唯一由程序"算"出的东西，只覆盖最朴素的一类失误：走完之后对方直接吃子就能净赚
（如象走到被兵控制的格、高价值子吃被保护的低价值子）。捉双、牵制等战术不在此列，仍由模型判断。
将杀只查一步杀（mate_in_one / allows_mate / mate_threat），只用来告诉模型"存在一步杀"，不提供具体着法。
算法是只含吃子的静止搜索：双方轮流吃子（不限于同一格），任何一方都可以随时停手，
所以对方吃了我一个子之后我方改吃别处更大的子，也会算进去；用合法着法生成，自动处理牵制、X 光和将军。
单格的静态交换（_exchange）仍用于"哪些子悬着"的判断（en_prise、双攻的目标）。

verify_line 用来核对模型为弃子给出的变化：按规则摆一遍，只报告是否合法、终点子力，不评价局面。
"""
import re
from functools import lru_cache

import chess

from .boardtext import COLOR_ZH, PIECE_VALUES, PIECE_ZH, display_san, material_points, parse_model_move
from .config import HANG_GUARD_MIN

_ORDER_VALUE = {**PIECE_VALUES, chess.KING: 100}  # 只用于"先用哪个子吃"的排序：王最后上
MAX_LINE_PLIES = 16
QS_MAX_PLIES = 12      # 吃子搜索最多连续吃几步
QS_MAX_NODES = 4000    # 一次搜索最多访问的局面数，超过就在当前分数上停手


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


def _capture_gain(board: chess.Board, move: chess.Move) -> int:
    """move 本身吃到 / 升变得到的分数（吃过路兵算 1，升变只算比兵多出的部分）。"""
    gain = 1 if board.is_en_passant(move) else _value(board.piece_at(move.to_square))
    if move.promotion:
        gain += PIECE_VALUES[move.promotion] - 1
    return gain


def _quiesce(board: chess.Board, alpha: int, beta: int, depth: int, nodes: list[int]) -> tuple[int, list[chess.Move]]:
    """轮到走的一方只用吃子（含升变）能净得多少分，任何时候都可以停手（不会小于 0）：(分数, 最佳吃子序列)。
    不限于同一格：对方吃了之后，我方可以吃回、也可以改吃别处更大的子（如对方吃我后，我先吃掉他的后）。
    alpha-beta 剪枝；nodes 是节点计数，超过 QS_MAX_NODES 就在当前分数上停手。"""
    nodes[0] += 1
    best, best_line = 0, []
    if depth == 0 or nodes[0] > QS_MAX_NODES or best >= beta:
        return best, best_line
    alpha = max(alpha, best)
    captures = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    captures.sort(key=lambda m: (-_capture_gain(board, m), _ORDER_VALUE[board.piece_type_at(m.from_square)]))
    for move in captures:
        gain = _capture_gain(board, move)
        board.push(move)
        reply, reply_line = _quiesce(board, gain - beta, gain - alpha, depth - 1, nodes)
        board.pop()
        if gain - reply > best:
            best, best_line = gain - reply, [move] + reply_line
            if best >= beta:
                break
            alpha = max(alpha, best)
    return best, best_line


def _best_capture(board: chess.Board) -> tuple[int, list[str], int | None]:
    """轮到走的一方只靠吃子最多能净得多少分：(净得, 吃子序列 SAN, 第一步吃子的落点格)。
    对方的应着是"吃回或改吃别处"里对他最好的，不限于同一格，所以对方吃子之后我方有更大的反吃时不算亏。"""
    if board.is_game_over():
        return 0, [], None
    work = board.copy(stack=False)
    gain, moves = _quiesce(work, -1000, 1000, QS_MAX_PLIES, [0])
    if gain <= 0 or not moves:
        return 0, [], None
    line = []
    for mv in moves:
        line.append(display_san(work, mv))
        work.push(mv)
    return gain, line, moves[0].to_square


def _passed(board: chess.Board) -> chess.Board:
    """假设轮到走的一方空一步，换对方走的局面（只用于"对方现在就能吃什么"的模拟）。"""
    after = board.copy(stack=False)
    after.push(chess.Move.null())
    return after


def en_prise(board: chess.Board) -> list[int]:
    """轮到走的一方的马、象、车、后里，对方现在就能在它的格上发起吃子交换净赚的那些格：
    被更低价值的子攻击（有保护也一样，对方吃一个、我吃回仍净亏）、没有保护、或攻击多于保护。"""
    me = board.turn
    opp = _passed(board)
    return [sq for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
            for sq in board.pieces(pt, me) if _exchange(opp, sq)[0] > 0]


@lru_cache(maxsize=8192)
def _fork_gain(fen: str) -> int:
    """fen 是某个双攻着法走完后的局面（轮到被攻击方走）。被攻击方可以走任何一步
    （逃子、吃掉攻击子、将军、反吃别处），攻击方之后只靠吃子，在最好的应对下仍能净赚多少分。
    能一步化解（≤ 0）的"双攻"不是真双攻。"""
    board = chess.Board(fen)
    best = None
    for reply in list(board.legal_moves):
        gain = _capture_gain(board, reply) if board.is_capture(reply) or reply.promotion else 0
        board.push(reply)
        won, _ = _quiesce(board, -1000, 1000, QS_MAX_PLIES, [0])
        board.pop()
        if best is None or gain - won > best:
            best = gain - won
    return -best if best is not None else 0


def double_attacks(board: chess.Board) -> list[tuple[chess.Move, list[int]]]:
    """轮到走的一方一步就能形成的双攻：走完后，走动的那个子同时打到对方两个以上"能净赚"的目标
    （王算一个目标，即将军；其余是我方在那格发起吃子交换能净赚的马象车后，兵不算），且走动的子本身不会被对方净赚吃掉。
    另外要求"化解不了"：对方走任何一步（逃子、吃掉攻击子、将军、反吃别处）之后，我方仅靠吃子仍能净赚 ≥ HANG_GUARD_MIN 分
    （_fork_gain）；能一步化解的不算。更深的反击（非吃子的威胁、将军连击）不在计算内，仍由模型判断。"""
    out = []
    for mv in board.legal_moves:
        after = board.copy(stack=False)
        after.push(mv)
        if after.is_checkmate() or _exchange(after, mv.to_square)[0] > 0:
            continue
        mine = _passed(after)  # 假设对方不应，我方再走一步能吃到什么
        targets = [sq for sq in chess.SquareSet(after.attacks(mv.to_square)) & after.occupied_co[after.turn]
                   if after.piece_type_at(sq) == chess.KING
                   or (after.piece_type_at(sq) != chess.PAWN and _exchange(mine, sq)[0] > 0)]
        if len(targets) >= 2 and _fork_gain(after.fen()) >= HANG_GUARD_MIN:
            out.append((mv, targets))
    return out


def format_double_attacks(board: chess.Board, items: list[tuple[chess.Move, list[int]]]) -> str:
    """double_attacks 的结果写成 "Ne6（同时攻击 后 d8、车 f8）；..."，SAN 以 board（走子前的局面）为准。"""
    return "；".join(f"{display_san(board, mv)}（同时攻击 "
                    + "、".join(f"{PIECE_ZH[board.piece_type_at(t)]} {chess.square_name(t)}" for t in targets)
                    + "）" for mv, targets in items)


def opp_double_attacks(board: chess.Board) -> tuple[chess.Board, list[tuple[chess.Move, list[int]]]]:
    """假设我方空走一步，对方一步可形成的双攻：(对方走子前的局面, 双攻列表)。被将军时空着不合法，返回空列表。"""
    if board.is_check():
        return board, []
    opp_board = _passed(board)
    return opp_board, double_attacks(opp_board)


def double_attack_text(board: chess.Board) -> str:
    """双方一步可形成的双攻，写成给模型看的事实；都没有返回空字符串。
    对方那一侧按"假设我方空走一步"计算，被将军时空着不合法，只列我方。"""
    lines = []
    mine = double_attacks(board)
    if mine:
        lines.append(f"- 我方一步可形成的双攻：{format_double_attacks(board, mine)}")
    opp_board, theirs = opp_double_attacks(board)
    if theirs:
        lines.append(f"- 对方下一步可形成的双攻（假设我方这步不管）：{format_double_attacks(opp_board, theirs)}")
    return "\n".join(lines)


def allowed_double_attacks(board: chess.Board, move: chess.Move) -> tuple[chess.Board, list]:
    """走 move 之后，对方一步可形成的双攻：(走完后的局面, 双攻列表)。"""
    after = board.copy(stack=False)
    after.push(move)
    return after, ([] if after.is_game_over() else double_attacks(after))


def fork_text(board: chess.Board, move: chess.Move, after: chess.Board, forks: list) -> str:
    """走 move 之后对方双攻的事实描述（after 为走完后的局面）。"""
    return f"走 {display_san(board, move)} 之后，对方一步可形成的双攻：{format_double_attacks(after, forks)}。"


def fork_safe_moves(board: chess.Board, min_loss: int) -> list[chess.Move]:
    """全部合法着法里，走完后对方没有双攻、同一格吃子交换也不会让我方净亏 ≥ min_loss 的那些。"""
    return [m for m in board.legal_moves
            if not allowed_double_attacks(board, m)[1] and material_risk(board, m)["loss"] < min_loss]


def material_risk(board: chess.Board, move: chess.Move) -> dict:
    """走 move 之后，对方只靠吃子（我方也只靠吃子应对）能让我方净亏多少（已扣除 move 本身吃到的子）。
    返回 {"loss": 净亏分数（≤0 表示不亏）, "line": 对方的最优吃子序列, "square": 对方第一步吃的格,
    "preexisting": 这个吃子在走 move 之前就已存在（同一个子、同样能净赚），move 没有解决它}。"""
    gained = _value(board.piece_at(move.to_square))
    if board.is_en_passant(move):
        gained = 1
    if move.promotion:
        gained += PIECE_VALUES[move.promotion] - 1
    after = board.copy(stack=False)
    after.push(move)
    best, best_line, best_sq = _best_capture(after)
    preexisting = (best_sq is not None and best > 0 and best_sq not in (move.from_square, move.to_square)
                   and _exchange(_passed(board), best_sq)[0] >= best)
    return {"loss": best - gained, "line": best_line,
            "square": chess.square_name(best_sq) if best_sq is not None else "", "preexisting": preexisting}


def _line_steps(board: chess.Board, move: chess.Move, line: list[str]) -> str:
    """把交换序列逐步写成"谁吃了谁"：Bxa1（对方象吃我方车）→ Rxa1（我方车吃回象）。"""
    me = board.turn
    b = board.copy(stack=False)
    b.push(move)
    steps = []
    for san in line:
        mv = parse_model_move(b, san)
        if mv is None:
            break
        mover, taken = b.piece_at(mv.from_square), b.piece_at(mv.to_square)
        who, whom = ("我方", "对方") if b.turn == me else ("对方", "我方")
        steps.append(f"{san}（{who}{PIECE_ZH[mover.piece_type]}吃{whom}{PIECE_ZH[taken.piece_type] if taken else '兵'}）")
        b.push(mv)
    return " → ".join(steps)


def risk_text(board: chess.Board, move: chess.Move, risk: dict) -> str:
    """把模拟结果写成给模型看的事实描述。"""
    san = display_san(board, move)
    steps = _line_steps(board, move, risk["line"]) or " ".join(risk["line"])
    pre = (f"这个吃子在走 {san} 之前就已存在，{san} 没有解决它。" if risk.get("preexisting") else "")
    return (f"走 {san} 之后，对方可以吃子 {risk['square']}：按双方只用吃子轮流应对、任何一方都可以随时停手来模拟，"
            f"最佳序列为 {steps}，我方净亏 {risk['loss']} 分"
            f"（兵1 马象3 车5 后9，已算上 {san} 本身吃到的子）。{pre}")


def mate_in_one(board: chess.Board) -> bool:
    """轮到走的一方有没有一步就将杀的着法。"""
    for mv in board.legal_moves:
        if board.gives_check(mv):
            board.push(mv)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                return True
    return False


def allows_mate(board: chess.Board, move: chess.Move) -> bool:
    """走 move 之后，对方有没有一步杀。"""
    after = board.copy(stack=False)
    after.push(move)
    return mate_in_one(after)


def mate_threat(board: chess.Board) -> bool:
    """假如现在轮到对方走（我方空一步），对方有没有一步杀。被将军时空着不合法，返回 False。"""
    if board.is_check():
        return False
    after = board.copy(stack=False)
    after.push(chess.Move.null())
    return mate_in_one(after)


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
