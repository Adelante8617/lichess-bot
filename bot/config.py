"""全部配置：从 .env / 环境变量读取，其他模块只从这里取值。"""
import json
import os

from dotenv import load_dotenv

load_dotenv()

LICHESS_TOKEN = os.getenv("LICHESS_TOKEN")

# ---- 下棋 / 复盘用的 LLM（任意 OpenAI 兼容接口），全部可用 .env 覆盖 ----
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("XIAOMI_API_KEY")
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
EMBED_API_KEY = os.getenv("EMBED_API_KEY") or os.getenv("OPENAI_API_KEY")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://api.qingyuntop.top/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_LOCAL_MODEL = os.getenv("EMBED_LOCAL_MODEL", "BAAI/bge-m3")  # 也可填本地目录
EMBED_DEVICE = os.getenv("EMBED_DEVICE") or None  # cuda / cpu，默认自动

_BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
EMBED_QUERY_PREFIX = os.getenv(
    "EMBED_QUERY_PREFIX",
    _BGE_ZH_QUERY_PREFIX if EMBED_BACKEND == "local" and "bge" in EMBED_LOCAL_MODEL.lower() else "")


# Stockfish：仅用于复盘分析，下棋阶段不调用
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "stockfish")
STOCKFISH_ANALYZE_DEPTH = int(os.getenv("STOCKFISH_DEPTH", "14"))

# 等待对局超时（秒），超时自动退出
WAIT_TIMEOUT_SEC = int(os.getenv("WAIT_TIMEOUT_SEC", "60"))

# 棋盘辅助（默认全关，关闭时行为与原来完全一致）。程序只提供原始关系与摆棋，不做任何判断
BOARD_RELATIONS = os.getenv("BOARD_RELATIONS", "0") == "1"   # prompt 附上子力关系图
ANALYSIS_BOARD = os.getenv("ANALYSIS_BOARD", "0") == "1"     # 提供 play_line 分析棋盘工具
PLAN_MEMORY = os.getenv("PLAN_MEMORY", "1") == "1"           # 把上一步定下的战略方针带给下一步
# 1 = 先用一次不思考的调用定下紧急情况 / 战略方针 / ≤3 个候选，再让决策调用只计算这几个候选。
# 思考模型的隐藏推理不受提示词约束，会逐个试合法着法；关掉第一步的思考才能强制"先定方针"
STRATEGY_STAGE = os.getenv("STRATEGY_STAGE", "0") == "1"
STRATEGY_STAGE_MAX_TOKENS = int(os.getenv("STRATEGY_STAGE_MAX_TOKENS", "2048"))
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
# 我方子力（兵1 马象3 车5 后9）领先 ≥ 该值时跳过复杂度判断，直接用 MATERIAL_LEAD_EFFORT（0 = 关闭）。
# 默认 12 ≈ 多一个后加一个轻子：领先这么多时稳妥简化就够，少于这个领先仍可能被翻盘，要正常思考
MATERIAL_LEAD_SKIP = int(os.getenv("MATERIAL_LEAD_SKIP", "12"))
MATERIAL_LEAD_EFFORT = os.getenv("MATERIAL_LEAD_EFFORT", "low").strip().lower()


# 每步自动召回的经验条数；设为 0 关闭自动召回（只保留模型主动调用 search_experience）
AUTO_RECALL_K = int(os.getenv("AUTO_RECALL_K", "3"))
SELF_CHECK_ROUNDS = int(os.getenv("SELF_CHECK_ROUNDS", "2"))   # 0 关闭落子前自检
# 丢子守卫：落子前按规则模拟对方的吃子交换（SEE），会净亏 ≥ HANG_GUARD_MIN 分时把模拟结果交给模型复查；
# 模型改选后再查新着法，最多 HANG_GUARD_ROUNDS 轮。轮数用完仍会丢子、模型又给不出拿回子力的具体变化时，
# 改用模型自己候选里不丢子的那个（没有则保持原着法）。这是对局中唯一由程序计算的检查
HANG_GUARD = os.getenv("HANG_GUARD", "1") == "1"
HANG_GUARD_MIN = int(os.getenv("HANG_GUARD_MIN", "2"))      # 默认 2：放过弃一兵的开局弃兵
HANG_GUARD_ROUNDS = int(os.getenv("HANG_GUARD_ROUNDS", "2"))
# 自检的思考档位上限：决策档位高于它时，自检降到该档（空 = 不封顶，沿用决策档位）
SELF_CHECK_EFFORT = os.getenv("SELF_CHECK_EFFORT", "low").strip().lower()


SNAPSHOT_OK_DELTA = 50        # 走子后己方评估损失 < 该值（cp）才算"好棋"，才允许入库
SNAPSHOT_DEDUPE_SIM = 0.95    # 与已有条目余弦相似度 ≥ 该值则视为重复，跳过
