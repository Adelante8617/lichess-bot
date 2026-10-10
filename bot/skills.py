"""技能（skill）：从过往对局提炼出的、按局面类型组织的做法，放在 skills/<name>/SKILL.md。

参照 Claude Code 的 skill：
- frontmatter 里的 name / description 常驻系统提示（技能目录），正文只在需要时加载（渐进式披露）；
- when：程序按规则判定的触发条件（局面阶段、子力、王的位置……），命中就把正文直接放进 user_prompt，
  不像 embedding 召回那样靠文本相似度；没有 when 的技能只能由模型调用 load_skill 读取；
- 技能是仓库里的普通文件，人可以直接读、改、用 git 审查；curate_skills.py 把赛后新增的教训归并进来。

触发条件只陈述局面事实（与守卫一样不替模型下结论），可用的键见 PREDICATES，详见 skills/README.md。
when 可以是一个条件表（全部满足才命中），也可以是条件表的列表（任一个命中即可）。
"""
import json
import os
import threading

import chess

from .boardtext import PIECE_VALUES, game_phase, material_lead
from .config import SKILL_AUTO_MAX, SKILL_STATS_PATH, SKILLS_DIR

LETTERS = {"P": chess.PAWN, "N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK, "Q": chess.QUEEN}


# ---------------- 局面事实 ----------------

def king_spot(board: chess.Board, color: bool) -> str:
    """王的位置：uncastled（还在初始格 e1/e8）/ kingside（g、h 线，己方前两横排）/
    queenside（a-c 线，己方前两横排）/ other。"""
    sq = board.king(color)
    if sq is None:
        return "other"
    rank = chess.square_rank(sq) if color == chess.WHITE else 7 - chess.square_rank(sq)
    file = chess.square_file(sq)
    if rank == 0 and file == 4:
        return "uncastled"
    if rank <= 1 and file >= 6:
        return "kingside"
    if rank <= 1 and file <= 2:
        return "queenside"
    return "other"


def last_move_captured(board: chess.Board) -> bool:
    """对方上一步是不是吃子（含吃过路兵）。"""
    if not board.move_stack:
        return False
    before = board.copy()
    move = before.pop()
    return before.is_capture(move)


def threatened(board: chess.Board) -> bool:
    """轮到走的一方有没有马、象、车、后正被攻击，且攻击者价值更低、或这个子没有保护。"""
    me = board.turn
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, me):
            attackers = board.attackers(not me, sq)
            if not attackers:
                continue
            cheapest = min(PIECE_VALUES.get(board.piece_type_at(a), 100) for a in attackers)
            if cheapest < PIECE_VALUES[pt] or not board.attackers(me, sq):
                return True
    return False


def passer_rank(board: chess.Board, color: bool) -> int:
    """color 一方最靠前的通路兵走到了第几横排（按该方视角，1-8）；没有通路兵为 0。"""
    best = 0
    for sq in board.pieces(chess.PAWN, color):
        file, rank = chess.square_file(sq), chess.square_rank(sq)
        ahead = range(rank + 1, 8) if color == chess.WHITE else range(0, rank)
        blocked = any(board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, not color)
                      for f in (file - 1, file, file + 1) if 0 <= f <= 7 for r in ahead)
        if not blocked:
            best = max(best, rank + 1 if color == chess.WHITE else 8 - rank)
    return best


def san_moves(board: chess.Board) -> str:
    """对局至今的着法，空格分隔的纯 SAN，不带回合号（"e4 e6 d4 d5"），供 opening 条件做前缀匹配。"""
    tmp = board.root()
    out = []
    for mv in board.move_stack:
        out.append(tmp.san(mv))
        tmp.push(mv)
    return " ".join(out)


def _pieces_on_board(board: chess.Board) -> set[str]:
    return {letter for letter, pt in LETTERS.items() if board.pieces(pt, chess.WHITE) or board.pieces(pt, chess.BLACK)}


def _as_list(v) -> list:
    return v if isinstance(v, list) else [v]


# 条件键 → 判定函数 (board, 条件值) -> bool。值可以是单个值或列表（列表表示"其中任一"）
PREDICATES = {
    "phase": lambda b, v: game_phase(b) in _as_list(v),                       # 开局 / 中局 / 残局
    "min_fullmove": lambda b, v: b.fullmove_number >= int(v),
    "max_fullmove": lambda b, v: b.fullmove_number <= int(v),
    "lead_min": lambda b, v: material_lead(b) >= int(v),                      # 我方子力分差（兵1 马象3 车5 后9）
    "lead_max": lambda b, v: material_lead(b) <= int(v),
    "only": lambda b, v: _pieces_on_board(b) <= set(_as_list(v)),             # 盘上（不计王）只有这些子种
    "has": lambda b, v: set(_as_list(v)) <= _pieces_on_board(b),              # 盘上必须有这些子种
    "queens": lambda b, v: bool(b.pieces(chess.QUEEN, chess.WHITE) or b.pieces(chess.QUEEN, chess.BLACK)) == bool(v),
    "my_king": lambda b, v: king_spot(b, b.turn) in _as_list(v),              # uncastled / kingside / queenside / other
    "opp_king": lambda b, v: king_spot(b, not b.turn) in _as_list(v),
    "in_check": lambda b, v: b.is_check() == bool(v),
    "opp_captured": lambda b, v: last_move_captured(b) == bool(v),            # 对方上一步吃了子
    "threatened": lambda b, v: threatened(b) == bool(v),                      # 我方有子被更低价值的子攻击或无保护被攻击
    "opp_passer_min": lambda b, v: passer_rank(b, not b.turn) >= int(v),      # 对方通路兵已到第几横排（对方视角）
    "my_passer_min": lambda b, v: passer_rank(b, b.turn) >= int(v),
    "my_color": lambda b, v: ("white" if b.turn == chess.WHITE else "black") in _as_list(v),
    "opening": lambda b, v: any((san_moves(b) + " ").startswith(p.strip() + " ") for p in _as_list(v)),
}


# 给整理技能的 LLM 看的条件说明（人看的版本在 skills/README.md）
PREDICATE_DOCS = {
    "phase": '"开局" / "中局" / "残局"，或它们的列表（前 10 回合为开局，双方轻重子总分 ≤26 为残局）',
    "min_fullmove / max_fullmove": "全回合数范围（整数）",
    "lead_min / lead_max": "我方子力分差（兵1 马象3 车5 后9）的范围（整数，负数表示落后）",
    "only": '盘上（不计王）只有这些子种，如 ["R", "P"]',
    "has": '盘上（任一方）必须有这些子种，如 ["Q"]',
    "queens": "true / false：盘上有没有后",
    "my_king / opp_king": '"uncastled"（还在初始格）/ "kingside" / "queenside" / "other"',
    "in_check": "true / false：我方正被将军",
    "opp_captured": "true / false：对方上一步吃了子",
    "threatened": "true / false：我方有马象车后被更低价值的子攻击、或被攻击且没有保护",
    "opp_passer_min / my_passer_min": "对方 / 我方最靠前的通路兵已到第几横排（整数 1-8，按该方视角）",
    "my_color": '"white" / "black"',
    "opening": '对局着法以其中之一开头（纯 SAN 前缀列表，如 ["e4 e6"]）',
}


# ---------------- SKILL.md 解析 ----------------

def _value(text: str):
    """frontmatter 的值：能按 JSON 解析的（数字、true/false、["a", "b"]）按 JSON，否则当字符串。"""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        return text.strip('"').strip("'")


def parse_skill(text: str) -> dict:
    """解析 SKILL.md：--- 包起来的 frontmatter（key: value；when: 下面缩进的 key: value，
    或缩进的 "- key: value" 开头表示一组新的条件）+ 正文。格式不对抛 ValueError。"""
    lines = text.replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        raise ValueError("缺少 frontmatter（第一行应为 ---）")
    try:
        end = lines.index("---", 1)
    except ValueError:
        raise ValueError("frontmatter 没有结束的 ---") from None
    meta: dict = {}
    groups: list[dict] | None = None
    for raw in lines[1:end]:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw[0] in " \t":
            if groups is None:
                raise ValueError(f"缩进行不在 when 下：{raw!r}")
            item = raw.strip()
            if item.startswith("- "):
                groups.append({})
                item = item[2:]
            if not groups:
                groups.append({})
            key, _, val = item.partition(":")
            groups[-1][key.strip()] = _value(val)
            continue
        key, _, val = raw.partition(":")
        key = key.strip()
        if key == "when" and not val.strip():
            groups = meta["when"] = []
        else:
            meta[key] = _value(val)
            groups = None
    for k in ("name", "description"):
        if not meta.get(k):
            raise ValueError(f"frontmatter 缺少 {k}")
    if isinstance(meta.get("when"), dict):  # 也允许一行写完：when: {"phase": "残局"}
        meta["when"] = [meta["when"]]
    if not isinstance(meta.get("when") or [], list) or not all(isinstance(g, dict) for g in meta.get("when") or []):
        raise ValueError("when 的格式不对")
    for group in meta.get("when") or []:
        unknown = set(group) - set(PREDICATES)
        if unknown:
            raise ValueError(f"未知的触发条件：{', '.join(sorted(unknown))}")
    return {"name": str(meta["name"]), "description": str(meta["description"]),
            "when": meta.get("when") or [], "priority": int(meta.get("priority", 0)),
            "body": "\n".join(lines[end + 1:]).strip()}


def load_skills(root: str | None = None) -> dict[str, dict]:
    """读取 root/<name>/SKILL.md，返回 {name: skill}。解析失败的跳过并打印原因。"""
    root = root or SKILLS_DIR
    skills: dict[str, dict] = {}
    if not os.path.isdir(root):
        return skills
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry, "SKILL.md")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                skill = parse_skill(f.read())
        except (OSError, ValueError) as e:
            print(f"[SKILL] 跳过 {path}：{e}")
            continue
        skill["path"] = path
        skills[skill["name"]] = skill
    return skills


_cache: dict[str, dict] | None = None


def all_skills() -> dict[str, dict]:
    """进程内只读一次；改了 SKILL.md 重启生效（或调用 reload()）。"""
    global _cache
    if _cache is None:
        _cache = load_skills()
    return _cache


def reload():
    global _cache
    _cache = None


# ---------------- 匹配与加载 ----------------

def matches(skill: dict, board: chess.Board) -> bool:
    return any(all(PREDICATES[k](board, v) for k, v in group.items()) for group in skill["when"])


def match_skills(board: chess.Board, skills: dict[str, dict] | None = None,
                 limit: int | None = None) -> list[dict]:
    """当前局面命中的技能，按 priority、条件数（越具体越先）排序，最多 limit 个（默认 SKILL_AUTO_MAX）。"""
    skills = all_skills() if skills is None else skills
    limit = SKILL_AUTO_MAX if limit is None else limit
    hit = [s for s in skills.values() if s["when"] and matches(s, board)]
    hit.sort(key=lambda s: (-s["priority"], -max(len(g) for g in s["when"]), s["name"]))
    return hit[:max(0, limit)]


def index_text(skills: dict[str, dict] | None = None) -> str:
    """技能目录（系统提示用）：每个技能一行 name：description。"""
    skills = all_skills() if skills is None else skills
    return "\n".join(f"- {s['name']}：{s['description']}" for s in skills.values())


def section_text(hits: list[dict]) -> str:
    """自动加载的技能正文（user_prompt 用）。"""
    return "\n\n".join(f"【{s['name']}】{s['description']}\n{s['body']}" for s in hits)


def load_skill_tool(name: str) -> str:
    skills = all_skills()
    skill = skills.get(str(name).strip())
    if skill is None:
        return f"没有名为 {name!r} 的技能。可用的技能：{', '.join(skills) or '（无）'}"
    return f"【{skill['name']}】{skill['description']}\n{skill['body']}"


# ---------------- 使用统计 ----------------

_stats_lock = threading.Lock()


def record_game(pgn_text: str, my_color: str, blunders: list[dict], path: str | None = None):
    """赛后统计：重放本局，记下我方每步走子前按规则命中了哪些技能，以及这些步里有几步被 Stockfish 判为 blunder。
    写入 data/skill_stats.json：{name: {"games": 局数, "moves": 命中步数, "blunders": 其中的 blunder 步数}}。
    只统计自动加载（规则命中）的技能；模型用 load_skill 主动读取的不在内。"""
    import io

    import chess.pgn

    path = path or SKILL_STATS_PATH
    skills = all_skills()
    if not skills:
        return
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return
    me = chess.WHITE if my_color == "白" else chess.BLACK
    bad_plies = {b["ply"] for b in blunders if b.get("side") == my_color}
    used: dict[str, list[int]] = {}
    board = game.board()
    for ply, move in enumerate(game.mainline_moves(), 1):
        if board.turn == me:
            for s in match_skills(board, skills):
                used.setdefault(s["name"], []).append(ply)
        board.push(move)
    with _stats_lock:
        try:
            with open(path, encoding="utf-8") as f:
                stats = json.load(f)
        except (OSError, ValueError):
            stats = {}
        for name, plies in used.items():
            st = stats.setdefault(name, {"games": 0, "moves": 0, "blunders": 0})
            st["games"] += 1
            st["moves"] += len(plies)
            st["blunders"] += len(bad_plies.intersection(plies))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
    if used:
        print("[SKILL] 本局命中：" + "，".join(f"{n}×{len(p)}" for n, p in used.items()))
