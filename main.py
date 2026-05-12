import os
import re
import io
import json
import time
import queue
import threading
import chess
import chess.pgn
import chess.engine
import berserk

from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

from logger_setup import setup_logger
from rag import RAGStore

# =========================
# LOGGING
# =========================
logger, LOG_PATH = setup_logger()
print(f"=== Lichess LLM Bot started, log file: {LOG_PATH} ===")

load_dotenv()

LICHESS_TOKEN = os.getenv("LICHESS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

session = berserk.TokenSession(LICHESS_TOKEN)
client = berserk.Client(session=session)

llm = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.qingyuntop.top/v1")

MODEL = "gpt-4.1"
EMBED_MODEL = "text-embedding-3-small"

# Stockfish：仅用于复盘分析，下棋阶段不调用
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "stockfish")
STOCKFISH_ANALYZE_DEPTH = int(os.getenv("STOCKFISH_DEPTH", "14"))

# 等待对局超时（秒），超时自动退出
WAIT_TIMEOUT_SEC = int(os.getenv("WAIT_TIMEOUT_SEC", "60"))

# =========================
# RAG: 开局库 + 经验记忆库
# =========================
def embed(text: str):
    resp = llm.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


opening_rag = RAGStore("data/openings.jsonl", embed)
experience_rag = RAGStore("data/experience.jsonl", embed)


def seed_openings_if_empty():
    """首次启动时给开局库塞一些基础开局思路，之后通过复盘自然增长。"""
    if len(opening_rag) > 0:
        return
    seeds = [
        ("Italian Game: 1.e4 e5 2.Nf3 Nc6 3.Bc4。快速出动轻子，向 f7 施压。",
         {"name": "Italian Game"}),
        ("Ruy Lopez: 1.e4 e5 2.Nf3 Nc6 3.Bb5。主教钉马形成长期压力。",
         {"name": "Ruy Lopez"}),
        ("Sicilian Defense: 1.e4 c5。黑方不对称反击，复杂战斗。",
         {"name": "Sicilian"}),
        ("French Defense: 1.e4 e6 2.d4 d5。黑方兵链稳固，注意 c8 象。",
         {"name": "French"}),
        ("Caro-Kann: 1.e4 c6 2.d4 d5。稳健，残局兵形好。",
         {"name": "Caro-Kann"}),
        ("Queen's Gambit: 1.d4 d5 2.c4。弃兵抢中心。",
         {"name": "Queen's Gambit"}),
        ("King's Indian Defense: 1.d4 Nf6 2.c4 g6。黑方让中心后反击。",
         {"name": "KID"}),
        ("English Opening: 1.c4。侧翼控 d5，灵活转位。",
         {"name": "English"}),
        ("London System: 1.d4 2.Nf3 3.Bf4。白方稳健体系，易掌握。",
         {"name": "London"}),
        ("Scandinavian: 1.e4 d5。黑方立刻挑战中心。",
         {"name": "Scandinavian"}),
        ("通用开局原则：抢中心、快速出动轻子、王车易位、不要早出皇后、不要重复走同一子。",
         {"name": "principles"}),
    ]
    print(f"[RAG] seeding {len(seeds)} openings ...")
    for text, meta in seeds:
        try:
            opening_rag.add(text, meta)
        except Exception as e:
            print(f"[RAG] seed failed: {e}")
            break


seed_openings_if_empty()
print(f"[RAG] openings={len(opening_rag)}  experience={len(experience_rag)}")


# =========================
# TOOLS（让模型自行决定是否调用）
# =========================
PLAY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_opening_book",
            "description": (
                "开局阶段（约前 10-15 步）可调用，根据局面描述检索开局思路。"
                "仅供参考，不强制采纳。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "对当前局面或想走开局的简短描述"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_experience",
            "description": "查询过往复盘经验，可在任意阶段调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]

# 复盘阶段额外可用工具：Stockfish 分析
REVIEW_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_experience",
            "description": "查询过往复盘经验，作为对比参考。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_with_stockfish",
            "description": (
                "调用 Stockfish 引擎分析刚刚结束的这盘棋，找出 blunder/mistake/inaccuracy。"
                "只用于复盘阶段。仅当你想验证哪些步失误、评估值如何变化时才调用。"
                "返回 JSON 字符串，包含每个关键节点的回合号、走法、评估变化、评语。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_points": {
                        "type": "integer",
                        "description": "最多返回多少个关键节点（默认 8）",
                        "default": 8
                    }
                }
            }
        }
    }
]


def run_tool(name: str, args: dict, ctx: dict | None = None) -> str:
    ctx = ctx or {}
    q = args.get("query", "")
    if name == "search_opening_book":
        hits = opening_rag.query(q, k=3)
        if not hits:
            return "（暂无相关记录）"
        return "\n".join(f"- [{h['score']:.2f}] {h['text']}" for h in hits)
    if name == "search_experience":
        hits = experience_rag.query(q, k=3)
        if not hits:
            return "（暂无相关记录）"
        return "\n".join(f"- [{h['score']:.2f}] {h['text']}" for h in hits)
    if name == "analyze_with_stockfish":
        pgn_text = ctx.get("pgn_text")
        my_color = ctx.get("my_color", "白")
        if not pgn_text:
            return "（无 PGN 可分析）"
        max_points = int(args.get("max_points", 8))
        return stockfish_analyze_pgn(pgn_text, my_color, max_points=max_points)
    return "unknown tool"


# =========================
# Stockfish 分析（仅复盘用）
# =========================
def _cp_from_score(score: chess.engine.PovScore, pov_white: bool) -> int:
    """统一从白方视角取 centipawn；mate 转成 ±100000。"""
    s = score.white() if pov_white else score.black()
    if s.is_mate():
        m = s.mate()
        return 100000 if (m is not None and m > 0) else -100000
    cp = s.score()
    return int(cp) if cp is not None else 0


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
    """收集双方所有 blunder（cp 损失 ≥ threshold_cp），含走子前后 FEN。
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
            try:
                info_b = engine.analyse(board, limit)
                cp_before = _cp_from_score(info_b["score"], pov_white=mover_is_white)
                best = info_b.get("pv", [None])[0]
                best_uci = best.uci() if best else ""
            except Exception as e:
                print(f"[STOCKFISH] analyse before failed: {e}")
                break
            board.push(move)
            fen_after = board.fen()
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
                    "best": best_uci,
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


# =========================
# LLM 决策：先思考再决定
# =========================
SYSTEM_PROMPT = """你是一个国际象棋 AI。每一步必须严格执行以下流程：

⚠️ 颜色与所有权（最重要）：
- user_prompt 会告知你执白还是执黑。盘面图中【大写=白方棋子(W*)，小写=黑方棋子(B*)】。
- 你只能移动【自己颜色】的棋子。move 的起点格必须是你方子力所在的格；若起点是对方棋子，立即视为严重错误。
- 在做 A 节观察时，请先在心里把"我方所有棋子位置"和"对方所有棋子位置"分别列清楚再分析。

A. 局面观察（必填，每项简短，可用列表/分号分隔）：
   - opp_intent: 对方上一步意图（≤40 字，开局第一手填空字符串）
   - my_attacked: 我方受到攻击/将被吃的棋子（格子+子力，可多个）
   - opp_attacked: 对方受到我攻击的棋子
   - my_hanging: 我方悬子（无保护或保护数<攻击数）
   - opp_hanging: 对方悬子
   - check_chance: 我方下一步可发动的将军（若有）
   - capture_chance: 我方可直接吃子的着法及目标价值
   - threats: 对方当前对我的最严重威胁（将杀线、双重攻击、吃要子等）
   - tactics: 一步内可实现的战术机会（捉双 fork / 钉子 pin / 串击 skewer / 抽将 discovered / 以小博大兑换 等）

B. 候选步评估：
   - candidates: 列出 3-5 个候选着法，每个包含 {"move": "e2e4", "pros": "...", "cons": "..."}
   - 对每个候选步必须考虑：走后对方是否有强力应对（吃子、将军、战术反击）使我损失过大或王不安全
   - 不要走入对方攻击范围导致丢子，除非有明确补偿

C. 经验对照：
   - 若过往经验库（可调用 search_experience）中存在类似失误模式，不要重复，除非在 think 中说明本次不同
   - 开局阶段可调用 search_opening_book
   - 注意：人类教练在对局过程中可能会在聊天框发送评价，但你在【对局期间无法看到】这些消息，
     这些指导仅在赛后复盘中提供给你。因此对局中请独立思考，不要等待或假设外部提示。

D. 综合思考与决策：
   - pv: 你预期的主变（principal variation），≥3 个 UCI 走法，格式 ["mymove","oppreply","myreply",...]。
     必须基于具体计算给出对方最强应招，体现你算了几步。残局/有强战术时 ≥5 步。
   - think: 综合分析。字数上限见 user_prompt 的「think 字数上限」字段，
     必须包含：为何选中、PV 中关键节点的计算、排除其他候选的具体原因（不是空泛口号）。
   - board_summary: ≤80 字。一句话刻画当前局面骨架（材料差、王安全、关键弱点、双方计划方向），
     供未来检索此类局面时复用，要客观、具象、可检索。
   - move: 最终唯一着法，必须来自给定的合法着法列表，UCI 格式；必须等于 pv[0]。

严格按以下 JSON 输出（不要 markdown 代码块，所有字段必须存在；找不到的项给空字符串或空数组）：
{
  "opp_intent": "",
  "my_attacked": "",
  "opp_attacked": "",
  "my_hanging": "",
  "opp_hanging": "",
  "check_chance": "",
  "capture_chance": "",
  "threats": "",
  "tactics": "",
  "candidates": [{"move":"e2e4","pros":"...","cons":"..."}],
  "pv": ["e2e4","e7e5","g1f3"],
  "board_summary": "",
  "think": "",
  "move": "e2e4"
}"""


def _render_board(board: chess.Board) -> str:
    """渲染带坐标的 ASCII 棋盘。
    白子用大写并加 'W:' 前缀的图例，黑子小写并加 'B:'，空格用 '.'。
    便于 LLM 一眼看清双方棋子分布，避免误用对方子力。"""
    piece_unicode_off = False  # 仍用 ASCII 字母，便于纯文本 LLM 阅读
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


def _piece_lists(board: chess.Board) -> tuple[str, str]:
    """返回 (白方子力描述, 黑方子力描述)，按子力类型分组并标格子。"""
    names = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
             chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}
    def list_for(color: bool) -> str:
        groups: dict[str, list[str]] = {v: [] for v in names.values()}
        for sq, p in board.piece_map().items():
            if p.color != color:
                continue
            groups[names[p.piece_type]].append(chess.square_name(sq))
        order = ["K", "Q", "R", "B", "N", "P"]
        parts = []
        for k in order:
            if groups[k]:
                parts.append(f"{k}:{','.join(sorted(groups[k]))}")
        return "  ".join(parts)
    return list_for(chess.WHITE), list_for(chess.BLACK)


def get_llm_move(board: chess.Board, ply: int, prev_board: chess.Board | None,
                 opp_last_move: str | None,
                 chat_messages: list | None = None):
    # 注意：教练在聊天框中的实时评价【只在赛后复盘】使用，
    # 对局进行中模型看不到，避免实时作弊式指导。chat_messages 仅作签名兼容。
    _ = chat_messages
    legal_moves = [m.uci() for m in board.legal_moves]
    if not legal_moves:
        return None, "no legal move", "", {}

    # 我方 / 对方 颜色字符串
    my_color_str = "白(WHITE, 大写字母)" if board.turn == chess.WHITE else "黑(BLACK, 小写字母)"
    opp_color_str = "黑(BLACK, 小写字母)" if board.turn == chess.WHITE else "白(WHITE, 大写字母)"
    white_pieces, black_pieces = _piece_lists(board)
    my_pieces = white_pieces if board.turn == chess.WHITE else black_pieces
    opp_pieces = black_pieces if board.turn == chess.WHITE else white_pieces

    board_ascii = _render_board(board)

    if prev_board is not None and opp_last_move:
        prev_ascii = _render_board(prev_board)
        prev_section = f"""【对方走子前局面】
FEN: {prev_board.fen()}
盘面(白=W*, 黑=B*, '.'=空):
{prev_ascii}

对方上一步(UCI): {opp_last_move}

【对方走子后 = 当前局面，轮到你走】
FEN: {board.fen()}
盘面(白=W*, 黑=B*, '.'=空):
{board_ascii}
（请对比两个盘面，明确对方此步的真实意图）"""
    else:
        prev_section = f"""【当前局面，轮到你走】（开局第一手，无对方上一步）
FEN: {board.fen()}
盘面(白=W*, 黑=B*, '.'=空):
{board_ascii}"""

    chat_section = ""

    # think 字数上限随阶段递增（开局靠模式识别，中残局需要更多算变空间）
    full_move = (ply + 1) // 2  # 1-based 回合
    if full_move <= 8:
        think_budget = 120
    elif full_move <= 16:
        think_budget = 300
    else:
        think_budget = 500

    user_prompt = f"""{prev_section}

==== 身份与子力（务必看清，不要走对方的子！）====
你执 {my_color_str}，你只能移动【你自己的子】。
我方({my_color_str}) 子力位置: {my_pieces}
对方({opp_color_str}) 子力位置: {opp_pieces}
盘面图例：每格两字符，首字母 W=白方 B=黑方，第二个字母为子种类(K/Q/R/B/N/P)，'.'=空。
合法走法已经替你过滤好——只列出了你能走的着，绝不会出现对方棋子的起点。

回合(ply): {ply}  全回合数(fullmove): {full_move}
轮到: {my_color_str}
合法走法: {legal_moves}{chat_section}

think 字数上限: {think_budget} 字
要求：
- 你输出的 move 起点格必须是上面"我方子力位置"列出过的格子，否则即非法；
- 必须给出 ≥3 步的 pv（含对方应招），用具体算变支撑判断；
- think 中至少出现一次"如果对方走 X 我就 Y"形式的分支讨论；
- board_summary 用 ≤80 字客观刻画当前局面骨架，便于以后向量检索。

请按系统提示输出完整 JSON。"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    think, move, opp_intent = "", "", ""
    obs: dict = {}

    MAX_ATTEMPTS = 3
    for attempt in range(1, MAX_ATTEMPTS + 1):
        content = ""
        for _ in range(3):  # 单轮内最多 3 次 tool 调用
            resp = llm.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=PLAY_TOOLS,
                temperature=0.7,
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                })
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    result = run_tool(tc.function.name, args)
                    print(f"[TOOL] {tc.function.name}({args}) -> {result[:120]}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue
            content = (msg.content or "").strip()
            # 把这条 assistant 消息加入历史，便于后续反馈
            messages.append({"role": "assistant", "content": content})
            break

        print(f"[LLM raw attempt={attempt}] {content}")

        cur_think, cur_move, cur_opp = "", "", ""
        cur_obs: dict = {}
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                cur_opp = str(obj.get("opp_intent", "")).strip()
                cur_think = str(obj.get("think", "")).strip()
                cur_move = str(obj.get("move", "")).strip()
                cur_obs = {k: obj.get(k, "") for k in
                           ["my_attacked", "opp_attacked", "my_hanging", "opp_hanging",
                            "check_chance", "capture_chance", "threats", "tactics",
                            "candidates", "pv", "board_summary"]}
            except Exception:
                pass
        if not cur_move:
            m2 = re.search(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", content)
            if m2:
                cur_move = m2.group(1)

        # 只要当前轮抓到了任何字段，就更新（即便最终走法非法，思考过程也保留最新）
        if cur_think:
            think = cur_think
        if cur_opp:
            opp_intent = cur_opp
        if cur_obs:
            obs = cur_obs

        if cur_move in legal_moves:
            move = cur_move
            return move, think, opp_intent, obs

        # 非法 / 缺失 → 给模型反馈，要求重想
        reason = "未给出 move 字段" if not cur_move else f"'{cur_move}' 不在合法走法列表中"
        print(f"[WARN] attempt {attempt}: illegal move ({reason})")
        if attempt < MAX_ATTEMPTS:
            messages.append({
                "role": "user",
                "content": (
                    f"你刚才输出的走法非法：{reason}。\n"
                    f"请重新思考。注意：必须从下面的合法走法列表中选一个，"
                    f"UCI 格式严格匹配（含可能的升变后缀如 q/r/b/n）：\n"
                    f"{legal_moves}\n"
                    f"再次按系统提示输出完整 JSON。"
                ),
            })
            continue

    # 三次都失败：fallback
    print(f"[WARN] all {MAX_ATTEMPTS} attempts illegal, fallback to first legal")
    move = legal_moves[0]
    return move, think, opp_intent, obs


# =========================
# 复盘：对局结束后让模型自己反思并存入经验库
# =========================
def post_game_review(pgn_text: str, result: str, my_color: str, move_log: list):
    print("[REVIEW] generating self-review ...")
    brief_moves = "\n".join(
        f"{p}. {mv}  思考:{(th or '')[:40]}"
        for p, _, mv, th in move_log[-40:]
    )
    system = """你正在复盘一盘刚下完的国际象棋。流程：
1. 先自行思考这盘棋的大致表现。
2. 如果你想验证具体哪些步走错了、引擎评估是多少，可调用 analyze_with_stockfish 工具（仅此时可用）。
3. 可选地调用 search_experience 看看是否和过往教训重合。
4. 最后严格输出 JSON（不要 markdown）：
{
  "summary": "总体不超过 150 字",
  "lessons": ["教训1（详细分析，≤100字）", "教训2", "教训3"]
}
每条 lesson 必须详细描述「局面特征 + 失误/正确决策 + 应对原则」三要素，便于未来向量检索复用，单条 ≤100 字，可以写到接近 100 字。"""

    user = f"""你执{my_color}，结果：{result}
PGN:
{pgn_text}

你每步的简短思考（最后 40 步）：
{brief_moves}
"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    ctx = {"pgn_text": pgn_text, "my_color": my_color}

    text = ""
    try:
        for _ in range(3):
            resp = llm.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=REVIEW_TOOLS,
                temperature=0.3,
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                })
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    result_tool = run_tool(tc.function.name, args, ctx=ctx)
                    print(f"[REVIEW-TOOL] {tc.function.name}({args}) -> {result_tool[:160]}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_tool,
                    })
                continue
            text = (msg.content or "").strip()
            break
    except Exception as e:
        print(f"[REVIEW] LLM call failed: {e}")
        return

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        print(f"[REVIEW] no JSON parsed. raw={text[:200]}")
        return
    try:
        obj = json.loads(m.group(0))
    except Exception as e:
        print(f"[REVIEW] JSON parse failed: {e}; raw={text[:200]}")
        return

    summary = obj.get("summary", "")
    lessons = obj.get("lessons", [])
    print(f"[REVIEW] summary: {summary}")
    meta = {"result": result, "color": my_color,
            "time": datetime.now().isoformat()}
    for ls in lessons:
        if isinstance(ls, str) and ls.strip():
            print(f"[REVIEW] +lesson: {ls}")
            experience_rag.add(ls.strip(), meta)
    experience_rag.add(f"[复盘] {summary}", meta)
    print(f"[REVIEW] experience size = {len(experience_rag)}")


def blunder_deep_review(pgn_text: str, result: str, my_color: str):
    """强制：用 Stockfish 找出本局所有 blunder（双方），
    对每个 blunder 让模型分析两个 FEN 的差异并给出新判断，
    再用 Stockfish 评估这个新判断，全部写入经验库。"""
    print("[BLUNDER-REVIEW] start ...")
    blunders = stockfish_collect_blunders(pgn_text, threshold_cp=200)
    if not blunders:
        print("[BLUNDER-REVIEW] no blunder found")
        return
    print(f"[BLUNDER-REVIEW] {len(blunders)} blunder(s) found")

    meta_base = {"result": result, "my_color": my_color,
                 "kind": "blunder", "time": datetime.now().isoformat()}

    for i, b in enumerate(blunders, 1):
        print(f"\n[BLUNDER {i}/{len(blunders)}] ply={b['ply']} {b['side']} "
              f"走了 {b['move']} (best={b['best']}) "
              f"cp {b['cp_before']} -> {b['cp_after']} (Δ={b['delta']})")

        prompt = f"""下面是本局中一个被 Stockfish 标记为 blunder 的关键节点。

阵营: {b['side']} 走子
回合(ply): {b['ply']}
实际走法: {b['move']}
引擎推荐: {b['best']}
评估变化(走子方视角, cp): {b['cp_before']} -> {b['cp_after']}  (Δ={b['delta']})

走子前 FEN: {b['fen_before']}
走子后 FEN: {b['fen_after']}

请：
1. 对比两个 FEN 的差异（哪些子动了、丢了什么、暴露了什么）。
2. 分析为什么这步是 blunder：忽视了什么威胁/战术？王安全/子力/兵形上有何问题？
3. 给出你认为「在走子前的局面下」更好的着法（your_better_move，UCI 格式），并简要说明理由。
4. 提取一条可复用的经验（lesson，描述局面特征 + 失误模式 + 应对原则，≤100 字）。

严格输出 JSON（不要 markdown）：
{{
  "diff": "≤80 字",
  "why_blunder": "≤120 字",
  "your_better_move": "e2e4",
  "your_reason": "≤80 字",
  "lesson": "≤100 字"
}}"""
        try:
            resp = llm.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            txt = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[BLUNDER-REVIEW] LLM failed: {e}")
            continue

        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            print(f"[BLUNDER-REVIEW] no JSON. raw={txt[:200]}")
            continue
        try:
            obj = json.loads(m.group(0))
        except Exception as e:
            print(f"[BLUNDER-REVIEW] JSON parse failed: {e}; raw={txt[:200]}")
            continue

        diff = obj.get("diff", "")
        why = obj.get("why_blunder", "")
        my_better = (obj.get("your_better_move") or "").strip()
        my_reason = obj.get("your_reason", "")
        lesson = obj.get("lesson", "").strip()

        print(f"[BLUNDER {i}] diff: {diff}")
        print(f"[BLUNDER {i}] why : {why}")
        print(f"[BLUNDER {i}] mine: {my_better}  reason: {my_reason}")

        # 用 Stockfish 评估模型给出的"更好走法"
        sf_eval = stockfish_eval_move(b["fen_before"], my_better) if my_better else \
                  {"error": "no move"}
        print(f"[BLUNDER {i}] sf_eval: {sf_eval}")

        meta = dict(meta_base)
        meta.update({"ply": b["ply"], "side": b["side"],
                     "actual_move": b["move"], "best": b["best"],
                     "delta": b["delta"]})

        # 写入主 lesson
        if lesson:
            entry = (f"[Blunder-Lesson] {lesson} "
                     f"(局面: ply{b['ply']} {b['side']}方走 {b['move']}, "
                     f"引擎推荐 {b['best']}, Δ={b['delta']}cp)")
            print(f"[BLUNDER {i}] +lesson: {entry[:120]}")
            experience_rag.add(entry, meta)

        # 写入模型对自身替代走法的评估
        if isinstance(sf_eval, dict) and "error" not in sf_eval:
            verdict = sf_eval.get("verdict", "?")
            entry2 = (
                f"[Blunder-AltMove] 局面 FEN={b['fen_before']} | "
                f"模型替代走法 {my_better} 理由: {my_reason} | "
                f"引擎评估: cp {sf_eval['cp_before']}->{sf_eval['cp_after']} "
                f"(Δ={sf_eval['delta']}, {verdict}); 引擎最佳 {sf_eval['best']}. "
                f"原失误走法 {b['move']} Δ={b['delta']}cp."
            )
            print(f"[BLUNDER {i}] +alt-eval: {entry2[:120]}")
            experience_rag.add(entry2, dict(meta, kind="blunder_alt_eval"))
        else:
            print(f"[BLUNDER {i}] alt eval skipped: {sf_eval}")

    print(f"[BLUNDER-REVIEW] done. experience size = {len(experience_rag)}")


def chat_review(chat_messages: list, result: str, my_color: str):
    """把本局聊天框中收到的指导整理为经验。读取 only。"""
    if not chat_messages:
        return
    # 过滤掉自己（理论上 bot 不发，但保险）
    lines = [c for c in chat_messages
             if c.get("username", "").lower() != my_username]
    if not lines:
        return
    print(f"[CHAT-REVIEW] {len(lines)} message(s) to summarize")
    joined = "\n".join(
        f"[ply={c.get('ply','?')} last={c.get('last_move') or '-'} "
        f"fen={c.get('fen','')}]\n"
        f"[{c.get('room','player')}] {c.get('username','')}: {c.get('text','')}"
        for c in lines
    )
    prompt = f"""本局({my_color}方, 结果 {result})期间收到的聊天指导（按时间顺序）：

{joined}

请从中提炼可复用的国际象棋经验/原则。
严格输出 JSON（不要 markdown）：
{{
  "summary": "≤120 字，本局教练给的整体方向",
  "lessons": ["≤100 字一条", "..."]
}}
lessons 数量 1~5 条；只保留有普适价值的内容，闲聊忽略。"""
    try:
        resp = llm.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        txt = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[CHAT-REVIEW] LLM failed: {e}")
        return
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        print(f"[CHAT-REVIEW] no JSON. raw={txt[:200]}")
        return
    try:
        obj = json.loads(m.group(0))
    except Exception as e:
        print(f"[CHAT-REVIEW] JSON parse failed: {e}")
        return
    summary = obj.get("summary", "").strip()
    lessons = obj.get("lessons", []) or []
    meta = {"result": result, "my_color": my_color,
            "kind": "chat_guidance", "time": datetime.now().isoformat()}
    if summary:
        experience_rag.add(f"[Chat-Summary] {summary}", meta)
        print(f"[CHAT-REVIEW] +summary: {summary[:120]}")
    for ls in lessons:
        if isinstance(ls, str) and ls.strip():
            experience_rag.add(f"[Chat-Lesson] {ls.strip()}", meta)
            print(f"[CHAT-REVIEW] +lesson: {ls.strip()[:120]}")
    # 同时把原始聊天存档供未来回溯
    experience_rag.add(
        f"[Chat-Raw] {joined[:800]}",
        dict(meta, kind="chat_raw"),
    )
    print(f"[CHAT-REVIEW] done. experience size = {len(experience_rag)}")


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


# =========================
# GAME LOOP
# =========================
print(f"Waiting for games ... (idle timeout: {WAIT_TIMEOUT_SEC}s)")
my_username = client.account.get()["username"].lower()
print(f"My username: {my_username}")


def _event_producer(q: "queue.Queue"):
    try:
        for ev in client.bots.stream_incoming_events():
            q.put(ev)
    except Exception as e:
        q.put({"__error__": str(e)})


event_queue: "queue.Queue" = queue.Queue()
threading.Thread(target=_event_producer, args=(event_queue,), daemon=True).start()

while True:
    try:
        event = event_queue.get(timeout=WAIT_TIMEOUT_SEC)
    except queue.Empty:
        print(f"No game in {WAIT_TIMEOUT_SEC}s, exit.")
        break

    if "__error__" in event:
        print(f"[ERROR] event stream: {event['__error__']}")
        break

    if event.get("type") != "gameStart":
        continue

    game_id = event["game"]["id"]
    print(f"=== Game started: {game_id} ===")

    is_white = None
    move_log = []
    last_moves_str = None
    chat_messages: list = []  # [{"username","text","room","time","fen","ply","last_move"}]
    current_fen = chess.STARTING_FEN
    current_ply = 0
    current_last_move = None

    try:
        for state in client.bots.stream_game_state(game_id):
            stype = state.get("type")

            if stype == "gameFull":
                white_name = state["white"].get("name", "").lower()
                is_white = (white_name == my_username)
                print(f"I play {'WHITE' if is_white else 'BLACK'}")
                moves_str = state["state"]["moves"]
                game_status = state["state"].get("status", "started")
                # gameFull 里也可能带已有 chatLines
                for c in state.get("chatLines", []) or []:
                    chat_messages.append({
                        "username": c.get("username", ""),
                        "text": c.get("text", ""),
                        "room": c.get("room", "player"),
                        "time": datetime.now().isoformat(),
                        "fen": current_fen,
                        "ply": current_ply,
                        "last_move": current_last_move,
                    })
                    print(f"[CHAT<<] [{c.get('room','player')}] "
                          f"{c.get('username','')}: {c.get('text','')}")
            elif stype == "gameState":
                moves_str = state["moves"]
                game_status = state.get("status", "started")
            elif stype == "chatLine":
                # Lichess 原生推送，无需轮询
                uname = state.get("username", "")
                text = state.get("text", "")
                room = state.get("room", "player")
                chat_messages.append({
                    "username": uname, "text": text, "room": room,
                    "time": datetime.now().isoformat(),
                    "fen": current_fen,
                    "ply": current_ply,
                    "last_move": current_last_move,
                })
                print(f"[CHAT<<] [{room}] {uname}: {text} "
                      f"(ply={current_ply}, last={current_last_move})")
                continue
            else:
                continue

            board = chess.Board()
            uci_list = moves_str.split() if moves_str else []
            for uci in uci_list:
                try:
                    board.push_uci(uci)
                except Exception:
                    pass

            # 更新"当前局面"快照，供之后的 chatLine 事件关联上下文
            current_fen = board.fen()
            current_ply = board.ply()
            current_last_move = uci_list[-1] if uci_list else None

            # 对局结束
            if game_status not in ("started", "created"):
                print(f"Game finished, status={game_status}")
                winner = state.get("winner")
                if winner == "white":
                    result = "1-0"
                elif winner == "black":
                    result = "0-1"
                else:
                    result = "1/2-1/2"
                pgn_text = build_pgn(uci_list, result)
                os.makedirs("games", exist_ok=True)
                pgn_path = os.path.join(
                    "games",
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{game_id}.pgn"
                )
                with open(pgn_path, "w", encoding="utf-8") as f:
                    f.write(pgn_text)
                print(f"PGN saved to {pgn_path}")

                my_color = "白" if is_white else "黑"
                post_game_review(pgn_text, result, my_color, move_log)
                try:
                    blunder_deep_review(pgn_text, result, my_color)
                except Exception as e:
                    print(f"[BLUNDER-REVIEW] failed: {e}")
                try:
                    chat_review(chat_messages, result, my_color)
                except Exception as e:
                    print(f"[CHAT-REVIEW] failed: {e}")
                break

            if is_white is None:
                continue

            my_turn = (board.turn == chess.WHITE and is_white) or \
                      (board.turn == chess.BLACK and not is_white)
            if not my_turn:
                continue

            if moves_str == last_moves_str:
                continue
            last_moves_str = moves_str

            print(f"\n--- ply {board.ply()+1} my move ---")
            print(board)

            # 构造「对方走子前」局面
            prev_board = None
            opp_last_move = None
            if uci_list:
                prev_board = chess.Board()
                for uci in uci_list[:-1]:
                    try:
                        prev_board.push_uci(uci)
                    except Exception:
                        pass
                opp_last_move = uci_list[-1]

            move, think, opp_intent, obs = get_llm_move(
                board, board.ply() + 1, prev_board, opp_last_move,
                chat_messages=chat_messages,
            )
            if opp_intent:
                print(f"[OPP]   {opp_intent}")
            if obs:
                for k, v in obs.items():
                    if v:
                        print(f"[OBS]   {k}: {str(v)[:200]}")
            print(f"[THINK] {think}")
            print(f"[MOVE]  {move}")

            try:
                client.bots.make_move(game_id, move)
                move_log.append((board.ply() + 1, board.fen(), move, think))
                print(f"Played: {move}")
            except Exception as e:
                print(f"Move failed: {e}")

            # 把当前盘面的 board_summary 即时写入经验库，
            # 让以后遇到相似局面时 search_experience 能召回。
            try:
                bs = (obs.get("board_summary") or "").strip() if obs else ""
                if bs:
                    pv = obs.get("pv") or []
                    pv_str = " ".join(pv) if isinstance(pv, list) else str(pv)
                    entry = (
                        f"[InGame-Snapshot] FEN={board.fen()} | "
                        f"摘要: {bs} | 选择: {move} | PV: {pv_str}"
                    )
                    experience_rag.add(entry, {
                        "kind": "in_game_snapshot",
                        "ply": board.ply() + 1,
                        "fullmove": (board.ply() + 2) // 2,
                        "time": datetime.now().isoformat(),
                    })
                    print(f"[SNAP] +{entry[:120]}")
            except Exception as e:
                print(f"[SNAP] failed: {e}")

            time.sleep(0.5)

    except Exception as e:
        print(f"[ERROR] game loop: {e}")
        continue

    print(f"Waiting for next game ... (idle timeout: {WAIT_TIMEOUT_SEC}s)")
