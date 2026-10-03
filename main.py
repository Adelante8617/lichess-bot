import os
import random
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
from live import live
import board_view

# =========================
# LOGGING
# =========================
logger, LOG_PATH = setup_logger()
print(f"=== Lichess LLM Bot started, log file: {LOG_PATH} ===")

load_dotenv()

LICHESS_TOKEN = os.getenv("LICHESS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XIAOMI_API_KEY = os.getenv("XIAOMI_API_KEY")

session = berserk.TokenSession(LICHESS_TOKEN)
client = berserk.Client(session=session)

# ---- 下棋 / 复盘用的 LLM（任意 OpenAI 兼容接口），全部可用 .env 覆盖 ----
LLM_API_KEY = os.getenv("LLM_API_KEY") or XIAOMI_API_KEY
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
MODEL = os.getenv("LLM_MODEL", "mimo-v2.5-pro")
# 附加请求体（供应商私有参数，原样合并进请求 JSON）。默认 {}：不传任何额外参数。
# 实测 micuapi 的 deepseek-v4-flash：不传参数时默认开启思考；传 {"thinking": {...}}（哪怕 type=enabled）
# 反而会关闭思考。需要更强思考可设 {"reasoning_effort": "high"}。
LLM_EXTRA_BODY = json.loads(os.getenv("LLM_EXTRA_BODY", "{}"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))       # 含思考过程的 token
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# ---- RAG 用的 embedding（换 embedding 模型后需重建向量：python reembed.py）----
# EMBED_BACKEND=api   → 走 OpenAI 兼容接口（含本地 Ollama / vLLM 等的 /v1 地址）
# EMBED_BACKEND=local → 进程内用 sentence-transformers 直接跑本地模型，无需任何网络
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "api").lower()
EMBED_API_KEY = os.getenv("EMBED_API_KEY") or OPENAI_API_KEY
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://api.qingyuntop.top/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_LOCAL_MODEL = os.getenv("EMBED_LOCAL_MODEL", "BAAI/bge-m3")  # 也可填本地目录
EMBED_DEVICE = os.getenv("EMBED_DEVICE") or None  # cuda / cpu，默认自动

_BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
EMBED_QUERY_PREFIX = os.getenv(
    "EMBED_QUERY_PREFIX",
    _BGE_ZH_QUERY_PREFIX if EMBED_BACKEND == "local" and "bge" in EMBED_LOCAL_MODEL.lower() else "")

llm = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
emb_model = (OpenAI(api_key=EMBED_API_KEY, base_url=EMBED_BASE_URL)
             if EMBED_BACKEND == "api" else None)
_local_embedder = None

# Stockfish：仅用于复盘分析，下棋阶段不调用
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "stockfish")
STOCKFISH_ANALYZE_DEPTH = int(os.getenv("STOCKFISH_DEPTH", "14"))

# 等待对局超时（秒），超时自动退出
WAIT_TIMEOUT_SEC = int(os.getenv("WAIT_TIMEOUT_SEC", "60"))

# 棋盘辅助（默认全关，关闭时行为与原来完全一致）。程序只提供原始关系与摆棋，不做任何判断
BOARD_RELATIONS = os.getenv("BOARD_RELATIONS", "0") == "1"   # prompt 附上子力关系图
ANALYSIS_BOARD = os.getenv("ANALYSIS_BOARD", "0") == "1"     # 提供 play_line 分析棋盘工具
PLAN_MEMORY = os.getenv("PLAN_MEMORY", "0") == "1"           # 跨步保留模型自己的长期计划
# 单步内最多几轮工具往返；开分析棋盘时默认放宽，便于多次摆变化
TOOL_ROUNDS = int(os.getenv("TOOL_ROUNDS", "10" if ANALYSIS_BOARD else "3"))

# ---- 开局快速模式：前 N 个全回合跳过自检/经验召回，用更省的请求参数 ----
# 被将军、或对方上一步吃了子时自动退回完整模式（按规则判断的事实，不评价好坏）
OPENING_FAST_MOVES = int(os.getenv("OPENING_FAST_MOVES", "0"))   # 0 = 关闭
# ---- 思考强度阶梯：首次用第一档，每次重试（思考被截断 / 走法非法）降一档 ----
# 档位：max / high / low → reasoning_effort=该档 + thinking enabled；off → thinking disabled；
# default → 不传任何思考参数（供应商默认）。如 THINK_LADDER=high,low,off
EFFORT_LEVELS = ("max", "high", "low")
THINK_LADDER = [x.strip().lower() for x in os.getenv("THINK_LADDER", "default").split(",") if x.strip()] \
    or ["default"]
# 开局快速模式使用的档位；若在 THINK_LADDER 中，重试时从它往下降
OPENING_EFFORT = os.getenv("OPENING_EFFORT", "off").strip().lower()
# 思考被截断、没给出答案时，带回思考末尾多少字给下一档（0 = 不带回）
TRUNCATE_REASONING_TAIL = int(os.getenv("TRUNCATE_REASONING_TAIL", "4000"))
# ---- 复杂度分流：每步先用一次不思考的独立调用判断复杂度，再按档位决定思考强度与 max_tokens ----
COMPLEXITY_CHECK = os.getenv("COMPLEXITY_CHECK", "1") == "1"   # 0 = 关闭，回到 THINK_LADDER 全阶梯
# 复杂度 → [起始档位, max_tokens]；重试时从起始档位沿 THINK_LADDER 往下降
COMPLEXITY_PROFILE = json.loads(os.getenv("COMPLEXITY_PROFILE") or
                                '{"simple": ["off", 4096], "medium": ["low", 8192], "complex": ["high", 16384]}')
COMPLEXITY_DEFAULT = os.getenv("COMPLEXITY_DEFAULT", "medium")  # 判断失败时使用
# 我方子力（兵1 马象3 车5 后9）领先 ≥ 该值时跳过复杂度判断，直接用 MATERIAL_LEAD_EFFORT（0 = 关闭）
MATERIAL_LEAD_SKIP = int(os.getenv("MATERIAL_LEAD_SKIP", "8"))
MATERIAL_LEAD_EFFORT = os.getenv("MATERIAL_LEAD_EFFORT", "low").strip().lower()

# =========================
# RAG: 开局库 + 经验记忆库
# =========================
def embed(text: str):
    if EMBED_BACKEND == "local":
        global _local_embedder
        if _local_embedder is None:
            from sentence_transformers import SentenceTransformer
            print(f"[EMBED] loading local model {EMBED_LOCAL_MODEL} (device={EMBED_DEVICE or 'auto'}) ...")
            _local_embedder = SentenceTransformer(EMBED_LOCAL_MODEL, device=EMBED_DEVICE)
        return _local_embedder.encode(text, normalize_embeddings=True).tolist()
    resp = emb_model.embeddings.create(model=EMBED_MODEL, input=text)
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

# 分析棋盘：模型自己摆变化，程序只执行规则，不评估、不推荐
PLAY_LINE_TOOL = {
    "type": "function",
    "function": {
        "name": "play_line",
        "description": (
            "分析棋盘：从当前局面出发，按顺序摆一串着法（双方交替，第一步是你的着法），"
            f"最多 {board_view.MAX_LINE_PLIES} 步。返回每步是否合法、是否吃子/将军，"
            "以及摆完后的棋盘、子力、合法着法和子力关系。"
            "它只执行规则，不评估局面、不给分数、不推荐着法。"
            "用来核对你计算的变化，避免多步之后看错棋盘；可多次调用比较不同分支。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "moves": {"type": "array", "items": {"type": "string"},
                          "description": "SAN 着法序列，如 [\"Bxh7+\", \"Kxh7\", \"Ng5+\"]"}
            },
            "required": ["moves"]
        }
    }
}


def play_tools() -> list:
    return PLAY_TOOLS + [PLAY_LINE_TOOL] if ANALYSIS_BOARD else PLAY_TOOLS


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


def extract_json(text: str) -> dict | None:
    """从模型输出里取出第一个能解析的 JSON 对象（容忍前后缀文字、多个花括号）。"""
    decoder = json.JSONDecoder()
    pos = text.find("{")
    while pos != -1:
        try:
            obj, _ = decoder.raw_decode(text, pos)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        pos = text.find("{", pos + 1)
    return None


def fallback_move(legal_moves: list[str]) -> str:
    """模型多次给出非法走法时的最后保底：随机选一个合法着法，
    不做任何局面判断（杀棋 / 吃子 / 安全性都不替模型看），只保证不因超时或非法着法判负。"""
    return random.choice(legal_moves)


def run_tool(name: str, args: dict, ctx: dict | None = None) -> str:
    ctx = ctx or {}
    q = args.get("query", "")
    if name == "search_opening_book":
        hits = opening_rag.query(EMBED_QUERY_PREFIX + q, k=3)
        if not hits:
            return "（暂无相关记录）"
        return "\n".join(f"- [{h['score']:.2f}] {h['text']}" for h in hits)
    if name == "search_experience":
        hits = experience_rag.query(EMBED_QUERY_PREFIX + q, k=3)
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
    if name == "play_line":
        board = ctx.get("board")
        if board is None:
            return "（当前没有可用的棋盘）"
        moves = args.get("moves") or []
        if isinstance(moves, str):  # 容忍模型传成 "e4 e5 Nf3"
            moves = moves.replace(",", " ").split()
        return board_view.play_line(board, moves)
    return "unknown tool"


# =========================
# Stockfish 分析（仅复盘用）
# =========================
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
            board_before = _render_board(board)
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
            board_after = _render_board(board)
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


# =========================
# LLM 决策：先思考再决定
# =========================
SYSTEM_PROMPT = """你是一个国际象棋 AI。每一步按以下流程思考，最后只输出一个 JSON。

⚠️ 颜色与所有权（最重要）：
- user_prompt 会告知你执白还是执黑。盘面图中 W*=白方棋子，B*=黑方棋子。
- 你只能移动【自己颜色】的棋子。合法走法列表已经替你过滤好，move 必须逐字取自该列表。
- 所有着法（candidates / pv / move）一律用标准代数记谱 SAN 书写，如 e4、Nf3、exd5、O-O、e8=Q。

0. 先判断局面复杂度，再决定思考投入（在思考的一开始就做，不要无脑长思考）：
   - simple：开局常规出子、只有一个明显应着（必须应将、必须吃回被兑的子）、双方没有子力接触。
     快速决定，简短核对一遍安全即可。
   - medium：常规中局，有若干合理计划但没有直接战术。比较 2-3 个候选，各算 2-3 步。
   - complex：存在吃子、将军、捉双、钉子、悬子、王翼攻击等子力接触，或残局需要精确计算。
     必须对每个认真考虑的候选逐一计算对方最强应招，直到局面平静。
   复杂度判断错了代价很大：只要盘面上存在任何可以吃子或将军的着法（包括对方的），就不是 simple。

A. 局面观察：
   - opp_intent: 对方上一步意图（≤40 字；开局第一手填空字符串；可结合完整着法历史判断对方整体计划）
   - my_attacked / opp_attacked: 我方 / 对方正受到攻击的棋子（格子+子力）
   - my_hanging / opp_hanging: 我方 / 对方的悬子（无保护，或攻击者多于保护者，或被价值更低的子攻击）
   - check_chance / capture_chance: 我方可发动的将军 / 可直接吃子的着法及目标价值
   - threats: 对方当前对我最严重的威胁（将杀、捉双、吃要子等）
   - tactics: 一步内可实现的战术机会（捉双 / 钉子 / 串击 / 闪击 / 以小换大 等）

B. 候选步评估：
   - candidates: 列出候选着法（simple 1-2 个，medium 2-3 个，complex 3-5 个），
     每个包含 {"move": "e4", "pros": "...", "cons": "..."}
   - 对每个候选：走后对方最强的应对是什么（吃子、将军、战术反击）？我会不会因此丢子或王不安全？

C. 行棋原则：
   - 吃子前必须确认：对方能不能吃回？吃回之后谁赚？（例如用象吃被兵保护的兵，就是用 3 换 1）
   - 不做未经验证的弃子：只有在算清楚能拿回子力、能将杀、或获得决定性优势时才弃子；
     "争取主动""打开线路""制造威胁"这类模糊理由不算。
   - 已经落后时，更要避免连续冒险；先稳住局面，不要孤注一掷。
   - 经验库：user_prompt 中可能附带自动召回的经验，仅供参考；也可调用 search_experience /
     search_opening_book 查询。与当前局面不符的经验直接忽略。
   - 人类教练的聊天评价只在赛后复盘中提供，对局中你看不到，请独立思考。

D. 落子前安全检查（必做；complex 局面要把检查过程写进 think）：
   想象你选定的着法已经走完，站在对方的角度检查：
   1) 我刚走的这个子，落点被对方哪些子攻击、被我方哪些子保护？
   2) 这步走完后，我方有没有其他子因此失去保护（原本被它保护的子、被闪开的线路）？
   3) 对方有没有将军、吃子、捉双、钉子等强力回应？对方最强回应之后我净得失多少子力？
   任何一项会让我白白丢子，就换一个候选重新检查。

E. 输出。字段顺序就是你的决策顺序：先复杂度，再观察、候选、推理，最后才是 pv 和 move。
   - complexity: "simple" / "medium" / "complex"；complexity_reason: ≤30 字
   - think: 推理总结。simple ≤80 字，medium ≤200 字，complex ≤400 字。
     写清为何选这步、关键变化、排除其他候选的具体原因，以及 D 节安全检查的结论。
   - pv: 由 think 的计算得出的主变，≥3 个 SAN，格式 ["我方着","对方应着","我方着",...]，pv[0] 必须等于 move。
   - board_summary: ≤80 字，客观刻画局面骨架（材料差、王安全、关键弱点、双方计划），供以后检索复用。
   - move: 最终唯一着法，逐字取自合法走法列表（SAN）。

严格按以下 JSON 输出（不要 markdown 代码块，所有字段必须存在；找不到的项给空字符串或空数组）：
{
  "complexity": "medium",
  "complexity_reason": "",
  "opp_intent": "",
  "my_attacked": "",
  "opp_attacked": "",
  "my_hanging": "",
  "opp_hanging": "",
  "check_chance": "",
  "capture_chance": "",
  "threats": "",
  "tactics": "",
  "candidates": [{"move":"e4","pros":"...","cons":"..."}],
  "think": "",
  "pv": ["e4","e5","Nf3"],
  "board_summary": "",
  "move": "e4"
}"""


def system_prompt() -> str:
    """棋盘辅助开关打开时，在原系统提示后追加 F 节说明；全关时与原提示完全相同。"""
    extra = []
    if BOARD_RELATIONS:
        extra.append(
            "- 【子力关系】由程序按规则精确列出：每个子控制的空格、攻击/保护了谁、被谁攻击/保护，"
            "直线上前后两子，王周边被控制的格，兵形。这些只是原始事实，保证不会看错，"
            "但它不会告诉你哪里有悬子、牵制、捉双，也不判断谁好谁坏——这些由你自己从关系中推理。"
            "A 节的观察项仍由你自己填写。")
    if ANALYSIS_BOARD:
        extra.append(
            f"- play_line 工具是一块分析棋盘：给它一串着法（第一步是你的），它按规则摆出来并返回"
            f"摆完后的局面与子力关系，不评估、不推荐。计算多步变化时（尤其 medium / complex），"
            f"先在脑中构思变化，再用它核对终点局面，确认没有看错；可多次调用比较不同分支，"
            f"并在变化的终点自己判断局面（子力、王的安全、双方的威胁）。"
            f"单步内工具往返最多 {TOOL_ROUNDS} 轮，simple 局面一般不必调用。")
    if PLAN_MEMORY:
        extra.append(
            "- 长期计划：JSON 中额外输出 \"plan\" 字段（≤60 字），写下接下来几步的战略计划"
            "（如\"f4-f5 在王翼进攻\"、\"换掉黑格象后占据 d5\"）。下一步你会在 user_prompt 中看到"
            "自己之前的计划：局面仍适用就继续执行，出现新情况（对方威胁、战术机会）就修改。"
            "战术优先于计划，不要为了执行计划而忽视眼前的危险。")
    if not extra:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + "\n\nF. 棋盘辅助与计划：\n" + "\n".join(extra)


# 模型自己的长期计划：{我方颜色: (定下计划时的着法序列 UCI, 计划)}。
# 新局面若不是在该序列基础上继续（换了一盘棋），计划自动作废。
_plans: dict[bool, tuple[list[str], str]] = {}


def recall_plan(board: chess.Board) -> str:
    stored = _plans.get(board.turn)
    if not stored:
        return ""
    moves, plan = stored
    now = [m.uci() for m in board.move_stack]
    return plan if now[:len(moves)] == moves else ""


def _render_board(board: chess.Board) -> str:
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


def _piece_lists(board: chess.Board) -> tuple[str, str]:
    """返回 (白方子力描述, 黑方子力描述)，按子种分组并标格子，如 '王 e1；后 d1；车 a1, h1'。"""
    def list_for(color: bool) -> str:
        parts = []
        for pt in (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
            sqs = sorted(chess.square_name(sq) for sq in board.pieces(pt, color))
            if sqs:
                parts.append(f"{PIECE_ZH[pt]} {', '.join(sqs)}")
        return "；".join(parts)
    return list_for(chess.WHITE), list_for(chess.BLACK)


def _strip_check(san: str) -> str:
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


OBS_KEYS = ["complexity", "complexity_reason", "my_attacked", "opp_attacked", "my_hanging",
            "opp_hanging", "check_chance", "capture_chance", "threats", "tactics",
            "candidates", "pv", "board_summary"]

# 每步自动召回的经验条数；设为 0 关闭自动召回（只保留模型主动调用 search_experience）
AUTO_RECALL_K = int(os.getenv("AUTO_RECALL_K", "3"))
SELF_CHECK_ROUNDS = int(os.getenv("SELF_CHECK_ROUNDS", "2"))   # 0 关闭落子前自检
# 自检的思考档位上限：决策档位高于它时，自检降到该档（空 = 不封顶，沿用决策档位）
SELF_CHECK_EFFORT = os.getenv("SELF_CHECK_EFFORT", "low").strip().lower()


def effort_kwargs(level: str) -> dict:
    """思考档位 → 请求参数（reasoning_effort + extra_body.thinking），与 LLM_EXTRA_BODY 合并。"""
    kw: dict = {}
    extra = dict(LLM_EXTRA_BODY)
    if level in EFFORT_LEVELS:
        kw["reasoning_effort"] = level
        extra["thinking"] = {"type": "enabled"}
    elif level == "off":
        extra["thinking"] = {"type": "disabled"}
    if extra:
        kw["extra_body"] = extra
    return kw


def think_ladder(start: str | None = None) -> list[str]:
    """本步使用的档位阶梯：从 start 开始沿 THINK_LADDER 往下降；start 不在阶梯中则只用它一档。"""
    if not start:
        return THINK_LADDER
    if start in THINK_LADDER:
        return THINK_LADDER[THINK_LADDER.index(start):]
    return [start]


def cap_ladder(levels: list[str], cap: str) -> list[str]:
    """把档位阶梯封顶到 cap：levels 起始档高于 cap（按 THINK_LADDER 顺序）时改从 cap 开始。"""
    if not cap or (cap in THINK_LADDER and levels and levels[0] in THINK_LADDER
                   and THINK_LADDER.index(levels[0]) >= THINK_LADDER.index(cap)):
        return levels
    return think_ladder(cap)


def llm_call(messages: list, tools: list | None = None, levels: list[str] | None = None,
             max_tokens: int | None = None):
    """下棋阶段统一的 LLM 调用，返回 (message, 实际使用的档位)。
    从 levels[0] 开始；若思考把 max_tokens 用光、正文为空，则带回思考末尾、换下一档直接要结论。"""
    levels = levels or THINK_LADDER[:1]
    max_tokens = max_tokens or LLM_MAX_TOKENS
    msgs, use_tools = messages, tools
    for i, level in enumerate(levels):
        kw = dict(model=MODEL, messages=msgs, temperature=LLM_TEMPERATURE,
                  max_tokens=max_tokens, **effort_kwargs(level))
        if use_tools:
            kw["tools"] = use_tools
        print(f"[LLM] effort={level}")
        choice = llm.chat.completions.create(**kw).choices[0]
        msg = choice.message
        if choice.finish_reason != "length":
            return msg, level
        print(f"[WARN] 模型输出达到 max_tokens={max_tokens} 被截断（effort={level}）")
        if (msg.content or "").strip() or msg.tool_calls or i == len(levels) - 1:
            return msg, level
        # 正文为空：带回思考末尾，降一档、不给工具，直接要最终 JSON
        reasoning = reasoning_of(msg)
        tail = reasoning[-TRUNCATE_REASONING_TAIL:] if TRUNCATE_REASONING_TAIL > 0 else ""
        note = (f"你刚才的思考过长被截断，没有给出最终答案。以下是你思考的最后部分：\n{tail}\n\n"
                if tail else "你刚才的思考过长被截断，没有给出最终答案。\n")
        msgs = messages + [{"role": "user", "content": note +
                            "不要再展开新的计算，直接根据已有分析选定着法，按系统提示输出完整 JSON。"}]
        use_tools = None
        print(f"[SALVAGE] 思考被截断（{len(reasoning)} 字），降档 {level} -> {levels[i + 1]} 直接要结论")
        live.stage(f"思考过长被截断，降档到 {levels[i + 1]}")
    return msg, level


def reasoning_of(msg) -> str:
    """思考模型返回的推理过程（deepseek 等放在 reasoning_content），没有则为空串。"""
    return (getattr(msg, "reasoning_content", None)
            or (getattr(msg, "model_extra", None) or {}).get("reasoning_content") or "")


def _material(board: chess.Board, color: bool) -> str:
    return "".join(f"{PIECE_ZH[pt]}{len(board.pieces(pt, color))}"
                   for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)
                   if board.pieces(pt, color)) or "仅剩王"


def _phase(board: chess.Board) -> str:
    if board.fullmove_number <= 10:
        return "开局"
    pieces = sum(len(board.pieces(pt, c)) * v
                 for pt, v in ((chess.QUEEN, 9), (chess.ROOK, 5), (chess.BISHOP, 3), (chess.KNIGHT, 3))
                 for c in (chess.WHITE, chess.BLACK))
    return "残局" if pieces <= 26 else "中局"


def _is_lesson(entry: dict) -> bool:
    """自动召回只取"教训"类条目：不含整局总结、局面快照、替代走法评估、聊天原文。"""
    return (entry["meta"].get("kind") in (None, "blunder", "chat_guidance")
            and not entry["text"].startswith(("[复盘]", "[Chat-Summary]")))


def recall_experience(board: chess.Board) -> list[dict]:
    """按"阶段 + 双方子力 + 最近着法"检索相关教训，直接放进 prompt。
    注意：这只是检索用的查询文本，不会展示给模型。"""
    if AUTO_RECALL_K <= 0:
        return []
    me = board.turn
    recent = " ".join(san_history(board).split()[-12:])
    q = (f"{_phase(board)}，我执{COLOR_ZH[me]}，我方子力 {_material(board, me)}，"
         f"对方子力 {_material(board, not me)}{'，正被将军' if board.is_check() else ''}；最近着法 {recent}")
    try:
        return experience_rag.query(EMBED_QUERY_PREFIX + q, k=AUTO_RECALL_K, filter_fn=_is_lesson)
    except Exception as e:
        print(f"[RECALL] failed: {e}")
        return []


def self_check(board: chess.Board, messages: list, move: chess.Move,
               legal_sans: list[str], complexity: str, levels: list[str] | None = None,
               max_tokens: int | None = None) -> tuple[chess.Move, list[dict]]:
    """落子前自检：让模型站在对方角度重新审视选定着法，发现会白丢子就换。
    只让模型自己复查，程序不做任何局面判断。返回 (最终着法, 每轮自检记录)。"""
    records = []
    current = move
    for rnd in range(1, SELF_CHECK_ROUNDS + 1):
        san = display_san(board, current)
        after = board.copy()
        after.push(current)
        opp = COLOR_ZH[after.turn]
        live.stage(f"第 {rnd} 轮自检：复查 {san}")
        relations = (f"\n子力关系（程序按规则列出的原始事实，从{opp}的视角）：\n"
                     f"{board_view.relations_text(after)}\n" if BOARD_RELATIONS else "")
        prompt = f"""在真正落子前做一次独立复查。你准备走 {san}。
不要沿用刚才的结论，重新看盘面。走完 {san} 之后的局面如下（轮到{opp}走）：
{_render_board(after)}
{relations}
请站在{opp}的角度，找出{opp}此时最强的应着，并回答：
1) 我刚走的子落点被对方哪些子攻击、被我方哪些子保护？
2) 这步是否让我方其他子失去保护？
3) 对方最强应着之后，我方净得失多少子力？
局面复杂度为 {complexity or "未知"}：simple 局面简短核对即可，complex 局面要认真计算。

如果 {san} 会白白丢子或导致严重后果，改选一个更好的着法（必须来自原合法走法列表）：
{", ".join(legal_sans)}

严格输出 JSON（不要 markdown）：
{{
  "opp_best_reply": "对方最强应着（SAN）",
  "danger": "走完后我方面临的具体危险，没有则写 无",
  "material_after": "对方最强应着后我方净得失，如 -3（丢马）/ 0 / +1",
  "verdict": "keep 或 change",
  "move": "keep 时填 {san}；change 时填新着法",
  "reason": "≤80 字"
}}"""
        msgs = messages + [{"role": "user", "content": prompt}]
        try:
            msg, _ = llm_call(msgs, levels=levels, max_tokens=max_tokens)
        except Exception as e:
            print(f"[SELF-CHECK] LLM failed: {e}")
            break
        content = (msg.content or "").strip()
        print(f"[SELF-CHECK round={rnd}] {content}")
        obj = extract_json(content) or {}
        rec = {"round": rnd, "checked": san,
               **{k: str(obj.get(k, "")) for k in
                  ("opp_best_reply", "danger", "material_after", "verdict", "move", "reason")},
               "reasoning": reasoning_of(msg)}
        records.append(rec)
        if str(obj.get("verdict", "")).strip().lower() != "change":
            break
        new = parse_model_move(board, str(obj.get("move", "")))
        if new is None or new == current:
            rec["note"] = "改选的着法无效或与原着法相同，保持原着法"
            print(f"[SELF-CHECK] {rec['note']}: {obj.get('move')!r}")
            break
        print(f"[SELF-CHECK] 改选 {san} -> {display_san(board, new)}")
        rec["changed_to"] = display_san(board, new)
        current = new
        # 进入下一轮时复查新着法；最后一轮的改选不再复查
    return current, records


def is_opening_fast(board: chess.Board, prev_board: chess.Board | None,
                    opp_last_move: str | None) -> bool:
    """开局快速模式：前 OPENING_FAST_MOVES 个全回合内，且未被将军、对方上一步不是吃子。"""
    if board.fullmove_number > OPENING_FAST_MOVES or board.is_check():
        return False
    if prev_board is not None and opp_last_move:
        try:
            if prev_board.is_capture(chess.Move.from_uci(opp_last_move)):
                return False
        except ValueError:
            pass
    return True


def material_lead(board: chess.Board) -> int:
    """轮到走棋一方的子力分差（兵1 马象3 车5 后9），正数表示我方领先。"""
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    return sum(v * (len(board.pieces(pt, board.turn)) - len(board.pieces(pt, not board.turn)))
               for pt, v in values.items())


def classify_complexity(board: chess.Board, last_section: str, legal_sans: list[str]) -> tuple[str, str]:
    """独立的一次不思考调用，只判断当前局面复杂度，返回 (simple/medium/complex, 理由)。
    程序只提供盘面与双方着法列表等原始事实，判断由模型做；失败时返回 COMPLEXITY_DEFAULT。"""
    opp_sans = "（我方正被将军，略）"
    if not board.is_check():
        tmp = board.copy()
        tmp.push(chess.Move.null())
        opp_sans = ", ".join(legal_san_map(tmp))
    recent = " ".join(san_history(board).split()[-12:]) or "（尚无着法）"
    prompt = f"""只判断当前局面的复杂度，不要选着、不要计算变化。
盘面(白=W*, 黑=B*, '.'=空，第二个字母为子种 K/Q/R/B/N/P)，轮到{COLOR_ZH[board.turn]}走：
{_render_board(board)}

对方刚走的一步：{last_section}
最近着法：{recent}
{board_meta(board)}

我方合法走法（带 + 为将军）：{", ".join(legal_sans)}
假如轮到对方走，对方的走法：{opp_sans}

分级标准：
- simple：常规出子/调动，双方都没有吃子或将军的着法；或只有一个明显应着（必须应将、必须吃回被兑的子）。
- medium：常规中局，有若干合理计划，双方子力有接触但没有直接战术。
- complex：存在吃子、将军、捉双、牵制、悬子、王翼攻击等直接战术，或残局需要精确计算。
只要双方任一方有吃子或将军的着法，就不是 simple。

严格输出 JSON（不要 markdown）：{{"complexity": "simple/medium/complex", "reason": "≤30 字"}}"""
    live.stage("判断局面复杂度")
    try:
        msg, _ = llm_call([{"role": "user", "content": prompt}], levels=["off"], max_tokens=512)
        obj = extract_json((msg.content or "").strip()) or {}
    except Exception as e:
        print(f"[COMPLEXITY] LLM failed: {e}")
        obj = {}
    level = str(obj.get("complexity", "")).strip().lower()
    reason = str(obj.get("reason", "")).strip()
    if level not in COMPLEXITY_PROFILE:
        print(f"[COMPLEXITY] 无法解析 {obj!r}，使用默认 {COMPLEXITY_DEFAULT}")
        return COMPLEXITY_DEFAULT, "复杂度判断失败，使用默认档"
    return level, reason


def get_llm_move(board: chess.Board, ply: int, prev_board: chess.Board | None,
                 opp_last_move: str | None,
                 chat_messages: list | None = None):
    # 注意：教练在聊天框中的实时评价【只在赛后复盘】使用，
    # 对局进行中模型看不到，避免实时作弊式指导。chat_messages 仅作签名兼容。
    _ = chat_messages
    san_map = legal_san_map(board)
    if not san_map:
        return None, "no legal move", "", {}
    legal_sans = list(san_map)

    # 我方 / 对方 颜色字符串
    my_color_str = COLOR_ZH[board.turn] + ("(WHITE)" if board.turn == chess.WHITE else "(BLACK)")
    white_pieces, black_pieces = _piece_lists(board)
    my_pieces = white_pieces if board.turn == chess.WHITE else black_pieces
    opp_pieces = black_pieces if board.turn == chess.WHITE else white_pieces

    if prev_board is not None and opp_last_move:
        last_section = describe_last_move(prev_board, opp_last_move)
    else:
        last_section = "（开局第一手，对方尚未走子）"

    live.thinking(ply)
    fast = is_opening_fast(board, prev_board, opp_last_move)
    # 本步复杂度 → 起始思考档位与 max_tokens
    complexity, complexity_reason = "", ""
    if fast:
        print(f"[FAST] 开局快速模式（fullmove={board.fullmove_number} ≤ {OPENING_FAST_MOVES}）")
        complexity, complexity_reason = "simple", "开局快速模式"
        ladder = think_ladder(OPENING_EFFORT)
        max_tokens = COMPLEXITY_PROFILE.get("simple", [None, LLM_MAX_TOKENS])[1]
    elif MATERIAL_LEAD_SKIP > 0 and (lead := material_lead(board)) >= MATERIAL_LEAD_SKIP:
        # 子力大幅领先：不调 LLM 判断复杂度，直接用较低档位；复杂度标签取起始档位相同的那一档
        complexity = next((k for k, (e, _) in COMPLEXITY_PROFILE.items() if e == MATERIAL_LEAD_EFFORT),
                          COMPLEXITY_DEFAULT)
        complexity_reason = f"我方子力领先 {lead} 分，跳过复杂度判断"
        ladder = think_ladder(MATERIAL_LEAD_EFFORT)
        max_tokens = COMPLEXITY_PROFILE.get(complexity, [None, LLM_MAX_TOKENS])[1]
        print(f"[COMPLEXITY] {complexity_reason} -> effort={MATERIAL_LEAD_EFFORT}, max_tokens={max_tokens}")
    elif COMPLEXITY_CHECK:
        complexity, complexity_reason = classify_complexity(board, last_section, legal_sans)
        start, max_tokens = COMPLEXITY_PROFILE[complexity]
        ladder = think_ladder(start)
        print(f"[COMPLEXITY] {complexity}（{complexity_reason}）-> effort={start}, max_tokens={max_tokens}")
    else:
        ladder, max_tokens = think_ladder(), LLM_MAX_TOKENS
    recalled = [] if fast else recall_experience(board)
    if recalled:
        recall_section = "\n".join(f"- {h['text']}" for h in recalled)
        print(f"[RECALL] {len(recalled)} 条: " + " | ".join(h["text"][:40] for h in recalled))
    else:
        recall_section = "（无）"

    full_move = (ply + 1) // 2  # 1-based 回合
    history = san_history(board) or "（尚无着法）"

    aid_section = ""
    if BOARD_RELATIONS:
        aid_section += ("\n==== 子力关系（程序按规则列出的原始事实，不含任何判断）====\n"
                        f"{board_view.relations_text(board)}\n")
    if PLAN_MEMORY:
        prev_plan = recall_plan(board)
        aid_section += ("\n==== 你之前定下的计划 ====\n"
                        f"{prev_plan or '（尚无，请根据局面制定）'}\n")

    user_prompt = f"""【当前局面，轮到你走】
盘面(白=W*, 黑=B*, '.'=空，第二个字母为子种 K/Q/R/B/N/P):
{_render_board(board)}

【对方刚走的一步】
{last_section}

【对局至今的全部着法（SAN）】
{history}
（可据此回顾双方计划、你自己此前的布局意图，并留意是否在重复局面）

==== 身份与子力 ====
你执 {my_color_str}，只能移动自己的子。
我方子力: {my_pieces}
对方子力: {opp_pieces}

==== 局面信息 ====
{board_meta(board)}
回合(ply): {ply}  全回合数(fullmove): {full_move}
{aid_section}
==== 经验库自动召回（仅供参考，与当前局面不符就忽略）====
{recall_section}

合法走法（SAN，已替你过滤，只含你能走的着；带 + 表示该着会将军）:
{", ".join(legal_sans)}

要求：
{"- 【开局快速模式】这是常规开局阶段：按开局原则（或调用 search_opening_book）快速选着，不要长时间计算；complexity 填 simple，think ≤60 字，pv 给 3 步即可；只需确认所走的子不会被白吃。" if fast else
  f"- 本步局面复杂度已单独判定为 {complexity}（{complexity_reason}），按此档决定思考投入（见系统提示第 0 节），complexity 字段照填 {complexity}；" if complexity else
  "- 先判断局面复杂度，按复杂度决定思考投入（见系统提示第 0 节）；"}
- 落子前完成系统提示 D 节的安全检查；
- move 必须逐字取自上面的合法走法列表，且等于 pv[0]；candidates / pv 全部用 SAN。

请按系统提示输出完整 JSON。"""

    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt},
    ]

    think, opp_intent, reasoning = "", "", ""
    obs: dict = {}
    warnings: list[str] = []
    recalled_view = [{"text": h["text"], "score": round(h["score"], 3)} for h in recalled]

    MAX_ATTEMPTS = 3
    step = 0  # 当前档位在 ladder 中的下标；被截断或走法非法都会往下降
    for attempt in range(1, MAX_ATTEMPTS + 1):
        content = ""
        live.stage(f"第 {attempt} 次决策" if attempt > 1 else "思考中")
        for rnd in range(TOOL_ROUNDS):  # 单轮内最多 TOOL_ROUNDS 次 tool 往返
            # 开分析棋盘时，最后一轮不再提供工具，逼模型给出最终答案
            last_round = ANALYSIS_BOARD and rnd == TOOL_ROUNDS - 1
            msg, level = llm_call(messages, tools=None if last_round else play_tools(),
                                  levels=ladder[step:], max_tokens=max_tokens)
            step = ladder.index(level)  # 若本轮因截断降了档，后续沿用降后的档位
            if msg.tool_calls:
                assistant = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                }
                if reasoning_of(msg):  # 思考模型在工具往返中需要带回推理过程
                    assistant["reasoning_content"] = reasoning_of(msg)
                messages.append(assistant)
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    result = run_tool(tc.function.name, args, {"board": board})
                    print(f"[TOOL] {tc.function.name}({args}) -> {result[:120]}")
                    live.tool(tc.function.name, args, result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue
            content = (msg.content or "").strip()
            reasoning = reasoning_of(msg) or reasoning
            # 把这条 assistant 消息加入历史，便于后续反馈 / 自检
            messages.append({"role": "assistant", "content": content})
            break

        if reasoning:
            print(f"[REASONING attempt={attempt}] {len(reasoning)} 字")
        print(f"[LLM raw attempt={attempt}] {content}")

        cur_think, cur_move, cur_opp = "", "", ""
        cur_obs: dict = {}
        obj = extract_json(content)
        if obj:
            cur_opp = str(obj.get("opp_intent", "")).strip()
            cur_think = str(obj.get("think", "")).strip()
            cur_move = str(obj.get("move", "")).strip()
            cur_obs = {k: obj.get(k, "") for k in OBS_KEYS}
            if complexity:  # 以独立判断的复杂度为准
                cur_obs["complexity"], cur_obs["complexity_reason"] = complexity, complexity_reason
            if PLAN_MEMORY:
                cur_obs["plan"] = str(obj.get("plan", "")).strip()
            pv = cur_obs.get("pv")
            if cur_move and isinstance(pv, list) and pv \
                    and _strip_check(str(pv[0])) != _strip_check(cur_move):
                print(f"[WARN] move={cur_move} 与 pv[0]={pv[0]} 不一致，以 move 为准")
                warnings.append(f"move={cur_move} 与 pv[0]={pv[0]} 不一致")

        # 只要当前轮抓到了任何字段，就更新（即便最终走法非法，思考过程也保留最新）
        if cur_think:
            think = cur_think
        if cur_opp:
            opp_intent = cur_opp
        if cur_obs:
            obs = cur_obs

        mv = parse_model_move(board, cur_move)
        if mv is not None:
            chosen_san = display_san(board, mv)
            checks: list[dict] = []
            if SELF_CHECK_ROUNDS > 0 and not fast:
                mv, checks = self_check(board, messages, mv, legal_sans, str(obs.get("complexity", "")),
                                        levels=cap_ladder(ladder[step:], SELF_CHECK_EFFORT),
                                        max_tokens=max_tokens)
                obs["self_check"] = checks
                if display_san(board, mv) != chosen_san:
                    warnings.append(f"自检后改选：{chosen_san} → {display_san(board, mv)}")
                    obs["pv"] = []  # 原主变基于旧着法，已失效
            if PLAN_MEMORY and obs.get("plan"):
                _plans[board.turn] = ([m.uci() for m in board.move_stack] + [mv.uci()], obs["plan"])
            live.decision(ply, move_san=display_san(board, mv), move_uci=mv.uci(),
                          first_choice=chosen_san, think=think, opp_intent=opp_intent, obs=obs,
                          reasoning=reasoning, recalled=recalled_view,
                          warnings=warnings + (["开局快速模式：已跳过自检"] if fast else []),
                          attempts=attempt, fallback=False)
            return mv.uci(), think, opp_intent, obs

        # 非法 / 缺失 → 给模型反馈，要求重想
        reason = "未给出 move 字段" if not cur_move else f"'{cur_move}' 不是当前局面的合法着法"
        print(f"[WARN] attempt {attempt}: illegal move ({reason})")
        warnings.append(f"第 {attempt} 次输出无效：{reason}")
        if attempt < MAX_ATTEMPTS:
            step = min(step + 1, len(ladder) - 1)
            # 只保留 system/user 初始提示 + 上一次回答，丢弃 tool 往返，防止重试时上下文膨胀
            messages = messages[:2] + [{"role": "assistant", "content": content}]
            messages.append({
                "role": "user",
                "content": (
                    f"你刚才输出的走法非法：{reason}。\n"
                    f"请重新思考。注意：必须从下面的合法走法列表中选一个（SAN 记谱，"
                    f"升变写法如 e8=Q）：\n"
                    f"{', '.join(legal_sans)}\n"
                    f"再次按系统提示输出完整 JSON。"
                ),
            })
            continue

    # 三次都失败：fallback
    move = fallback_move([m.uci() for m in san_map.values()])
    print(f"[WARN] all {MAX_ATTEMPTS} attempts illegal, random legal fallback -> {move}")
    live.decision(ply, move_san=display_san(board, chess.Move.from_uci(move)), move_uci=move,
                  think=think, opp_intent=opp_intent, obs=obs, reasoning=reasoning,
                  recalled=recalled_view,
                  warnings=warnings + ["三次均无效，随机选择合法着法保底"],
                  attempts=MAX_ATTEMPTS, fallback=True)
    return move, think, opp_intent, obs


# =========================
# 复盘：对局结束后让模型自己反思并存入经验库
# =========================
def post_game_review(pgn_text: str, result: str, my_color: str, move_log: list):
    print("[REVIEW] generating self-review ...")
    brief_moves = "\n".join(
        f"ply{p}. {uci_to_san(fen, mv)}  思考:{(th or '')[:40]}"
        for p, fen, mv, th in move_log[-40:]
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

    obj = extract_json(text)
    if obj is None:
        print(f"[REVIEW] no JSON parsed. raw={text[:200]}")
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
              f"走了 {b['san']} (best={b['best_san']}) "
              f"cp {b['cp_before']} -> {b['cp_after']} (Δ={b['delta']})")

        prompt = f"""下面是本局中一个被 Stockfish 标记为 blunder 的关键节点。

阵营: {b['side']} 走子
回合(ply): {b['ply']}
此前的全部着法（SAN）: {b['history'] or '（开局）'}
实际走法: {b['san']}
引擎推荐: {b['best_san']}
评估变化(走子方视角, cp): {b['cp_before']} -> {b['cp_after']}  (Δ={b['delta']})

走子前盘面(白=W*, 黑=B*, '.'=空):
{b['board_before']}

走子后盘面:
{b['board_after']}

请：
1. 对比两个盘面的差异（哪些子动了、丢了什么、暴露了什么）。
2. 分析为什么这步是 blunder：忽视了什么威胁/战术？王安全/子力/兵形上有何问题？
3. 给出你认为「在走子前的局面下」更好的着法（your_better_move，SAN 记谱，如 Nf3），并简要说明理由。
4. 提取一条可复用的经验（lesson，描述局面特征 + 失误模式 + 应对原则，≤100 字）。

严格输出 JSON（不要 markdown）：
{{
  "diff": "≤80 字",
  "why_blunder": "≤120 字",
  "your_better_move": "e4",
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

        obj = extract_json(txt)
        if obj is None:
            print(f"[BLUNDER-REVIEW] no JSON. raw={txt[:200]}")
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
        alt_mv = parse_model_move(chess.Board(b["fen_before"]), my_better) if my_better else None
        sf_eval = stockfish_eval_move(b["fen_before"], alt_mv.uci()) if alt_mv else \
                  {"error": f"无法解析或非法的着法: {my_better!r}"}
        print(f"[BLUNDER {i}] sf_eval: {sf_eval}")

        meta = dict(meta_base)
        meta.update({"ply": b["ply"], "side": b["side"],
                     "actual_move": b["san"], "best": b["best_san"],
                     "delta": b["delta"]})

        # 写入主 lesson
        if lesson:
            entry = (f"[Blunder-Lesson] {lesson} "
                     f"(局面: ply{b['ply']} {b['side']}方走 {b['san']}, "
                     f"引擎推荐 {b['best_san']}, Δ={b['delta']}cp)")
            print(f"[BLUNDER {i}] +lesson: {entry[:120]}")
            experience_rag.add(entry, meta)

        # 写入模型对自身替代走法的评估
        if isinstance(sf_eval, dict) and "error" not in sf_eval:
            verdict = sf_eval.get("verdict", "?")
            entry2 = (
                f"[Blunder-AltMove] 走法历史: {b['history'] or '（开局）'} | "
                f"模型替代走法 {my_better} 理由: {my_reason} | "
                f"引擎评估: cp {sf_eval['cp_before']}->{sf_eval['cp_after']} "
                f"(Δ={sf_eval['delta']}, {verdict}); "
                f"引擎最佳 {uci_to_san(b['fen_before'], sf_eval['best'])}. "
                f"原失误走法 {b['san']} Δ={b['delta']}cp."
            )
            print(f"[BLUNDER {i}] +alt-eval: {entry2[:120]}")
            experience_rag.add(entry2, dict(meta, kind="blunder_alt_eval"))
        else:
            print(f"[BLUNDER {i}] alt eval skipped: {sf_eval}")

    print(f"[BLUNDER-REVIEW] done. experience size = {len(experience_rag)}")


def chat_review(chat_messages: list, result: str, my_color: str, my_username: str):
    """把本局聊天框中收到的指导整理为经验。读取 only。"""
    if not chat_messages:
        return
    # 过滤掉自己（理论上 bot 不发，但保险）
    lines = [c for c in chat_messages
             if c.get("username", "").lower() != my_username.lower()]
    if not lines:
        return
    print(f"[CHAT-REVIEW] {len(lines)} message(s) to summarize")
    joined = "\n".join(
        f"[ply={c.get('ply','?')} 此前着法: {c.get('history') or '（开局）'}]\n"
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
    obj = extract_json(txt)
    if obj is None:
        print(f"[CHAT-REVIEW] no JSON. raw={txt[:200]}")
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


SNAPSHOT_OK_DELTA = 50        # 走子后己方评估损失 < 该值（cp）才算"好棋"，才允许入库
SNAPSHOT_DEDUPE_SIM = 0.95    # 与已有条目余弦相似度 ≥ 该值则视为重复，跳过


def record_snapshot(snapshots: list, board: chess.Board, move: str, obs: dict):
    """对局中缓存一条快照（走子前 FEN + 模型自己的摘要/PV），不写库。"""
    bs = (obs.get("board_summary") or "").strip() if obs else ""
    if not bs:
        return
    pv = obs.get("pv") or []
    snapshots.append({
        "fen": board.fen(),  # 仅供赛后 Stockfish 验证，不写入经验库文本
        "history": san_history(board),
        "ply": board.ply() + 1,
        "move": move,
        "san": uci_to_san(board.fen(), move),
        "summary": bs,
        "pv": " ".join(map(str, pv)) if isinstance(pv, list) else str(pv),
    })


def commit_verified_snapshots(snapshots: list, result: str, my_color: str):
    """赛后用 Stockfish 验证每条快照里模型选的着法，只有"好棋"才写入经验库，
    避免把模型自己的错误判断当成经验反复召回。同时对相近条目去重。"""
    if not snapshots:
        return
    kept = skipped_bad = skipped_dup = 0
    for s in snapshots:
        ev = stockfish_eval_move(s["fen"], s["move"])
        if "error" in ev or ev.get("delta", 10**9) >= SNAPSHOT_OK_DELTA:
            skipped_bad += 1
            continue
        entry = (f"[Verified-Snapshot] 走法历史: {s['history'] or '（开局）'} | "
                 f"摘要: {s['summary']} | "
                 f"选择: {s['san']}（Stockfish 验证 Δ={ev['delta']}cp，"
                 f"引擎最佳 {uci_to_san(s['fen'], ev['best'])}） | PV: {s['pv']}")
        wrote = experience_rag.add(entry, {
            "kind": "verified_snapshot", "verified": True,
            "ply": s["ply"], "result": result, "color": my_color,
            "delta": ev["delta"], "time": datetime.now().isoformat(),
        }, dedupe_threshold=SNAPSHOT_DEDUPE_SIM)
        if wrote:
            kept += 1
        else:
            skipped_dup += 1
    print(f"[SNAP] {len(snapshots)} 条快照: 写入 {kept}, 未通过验证 {skipped_bad}, 重复 {skipped_dup}")


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


# =========================
# GAME LOOP
# =========================

def run_lichess():
    print(f"Waiting for games ... (idle timeout: {WAIT_TIMEOUT_SEC}s)")
    my_username = client.account.get()["username"].lower()
    print(f"My username: {my_username}")


    def _event_producer(q: "queue.Queue"):
        try:
            for ev in client.bots.stream_incoming_events():
                q.put(ev)
        except Exception as e:
            q.put({"__error__": str(e)})


    MAX_STREAM_RETRIES = 6


    def resilient_game_stream(game_id: str):
        """包装 stream_game_state：断线 / 流意外关闭时指数退避重连，
        重连后 Lichess 会重发 gameFull，主循环据此恢复局面。收到终局状态才结束。"""
        retries = 0
        while True:
            try:
                for state in client.bots.stream_game_state(game_id):
                    retries = 0
                    yield state
                    status = state.get("status") or (state.get("state") or {}).get("status", "started")
                    if status not in ("started", "created"):
                        return
                print(f"[STREAM] game {game_id} stream closed before finish, reconnecting")
            except Exception as e:
                print(f"[STREAM] game {game_id} stream error: {e}")
            retries += 1
            if retries > MAX_STREAM_RETRIES:
                raise RuntimeError(f"game stream lost after {MAX_STREAM_RETRIES} reconnects")
            wait = min(2 ** retries, 20)
            print(f"[STREAM] reconnect in {wait}s ({retries}/{MAX_STREAM_RETRIES})")
            time.sleep(wait)


    def handle_challenge(ev: dict):
        """自动接受标准规则的挑战，其余拒绝。"""
        ch = ev.get("challenge", {})
        cid = ch.get("id")
        challenger = (ch.get("challenger") or {}).get("name", "?")
        variant = (ch.get("variant") or {}).get("key", "standard")
        try:
            if variant != "standard":
                client.bots.decline_challenge(cid, reason="standard")
                print(f"[CHALLENGE] declined {cid} from {challenger} (variant={variant})")
            else:
                client.bots.accept_challenge(cid)
                print(f"[CHALLENGE] accepted {cid} from {challenger}")
        except Exception as e:
            print(f"[CHALLENGE] handle {cid} failed: {e}")


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

        if event.get("type") == "challenge":
            handle_challenge(event)
            continue

        if event.get("type") != "gameStart":
            continue

        game_id = event["game"]["id"]
        print(f"=== Game started: {game_id} ===")

        is_white = None
        move_log = []
        snapshots: list = []  # 本局待验证的局面快照
        last_moves_str = None
        chat_messages: list = []  # [{"username","text","room","time","fen","ply","last_move"}]
        current_fen = chess.STARTING_FEN
        current_history = ""  # 当前局面之前的 SAN 走法，给聊天复盘用
        current_ply = 0
        current_last_move = None

        try:
            for state in resilient_game_stream(game_id):
                stype = state.get("type")

                if stype == "gameFull":
                    last_moves_str = None  # 重连后会重发 gameFull，允许重新决策当前手
                    white_name = state["white"].get("name", "").lower()
                    is_white = (white_name == my_username)
                    print(f"I play {'WHITE' if is_white else 'BLACK'}")
                    opp = state["black" if is_white else "white"]
                    live.start_game(game_id, "lichess", "白" if is_white else "黑",
                                    opp.get("name") or opp.get("id") or "对手")
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
                            "history": current_history,
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
                            "history": current_history,
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
                current_history = san_history(board)
                live.sync(uci_list)
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
                    live.set_status("reviewing", result)
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
                        chat_review(chat_messages, result, my_color, my_username)
                    except Exception as e:
                        print(f"[CHAT-REVIEW] failed: {e}")
                    try:
                        commit_verified_snapshots(snapshots, result, my_color)
                    except Exception as e:
                        print(f"[SNAP] commit failed: {e}")
                    live.set_status("finished", result)
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

                try:
                    move, think, opp_intent, obs = get_llm_move(
                        board, board.ply() + 1, prev_board, opp_last_move,
                        chat_messages=chat_messages,
                    )
                except Exception as e:
                    # LLM 接口异常也不能让这盘棋超时判负
                    print(f"[ERROR] get_llm_move failed: {e}, random legal fallback")
                    legal = [m.uci() for m in board.legal_moves]
                    move, think, opp_intent, obs = fallback_move(legal), "", "", {}
                    live.decision(board.ply() + 1, move_san=display_san(board, chess.Move.from_uci(move)),
                                  move_uci=move, think="", opp_intent="", obs={},
                                  warnings=[f"LLM 调用失败：{e}", "随机选择合法着法保底"],
                                  attempts=0, fallback=True)
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

                # 局面快照只先缓存在内存，赛后经 Stockfish 验证再决定是否写入经验库
                record_snapshot(snapshots, board, move, obs)

                time.sleep(0.5)

        except Exception as e:
            print(f"[ERROR] game loop: {e}")
            continue

        print(f"Waiting for next game ... (idle timeout: {WAIT_TIMEOUT_SEC}s)")


if __name__ == "__main__":
    run_lichess()
