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
# 1 = 流式请求：边生成边返回，长思考不会被中转站网关（如 Cloudflare 524，约 2 分钟收不到响应就断）掐断；
# 程序把所有片段拼回成完整响应（正文、思考、工具调用、结束原因等），调用方与非流式无区别
LLM_STREAM = os.getenv("LLM_STREAM", "1") == "1"
# 读超时（秒）：流式时是两个片段之间最长的等待，非流式时是等整个响应
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))
# 还没收到任何数据就失败（连接错误 / 超时 / 429 / 5xx）时重试几次；SDK 自带的重试关掉，由程序控制
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "1"))

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
# 把上一步算出的主变 pv 及其目的带给下一步：对方按主变应着时续走（连杀 / 对方唯一应着时不调 LLM 直接走），
# 对方偏离主变时告诉模型"预期 vs 实际"
PV_MEMORY = os.getenv("PV_MEMORY", "1") == "1"
PV_FOLLOW_EFFORT = os.getenv("PV_FOLLOW_EFFORT", "low").strip().lower()  # 按主变续走、需要模型确认时的档位
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
# 思考被截断、没给出答案时怎么把已有分析交给下一档：
# summary = 先用一次不思考的独立调用把整段思考压缩成要点；tail = 直接带回思考末尾；none = 不带回
TRUNCATE_SALVAGE = os.getenv("TRUNCATE_SALVAGE", "summary").strip().lower()
TRUNCATE_SUMMARY_MAX_TOKENS = int(os.getenv("TRUNCATE_SUMMARY_MAX_TOKENS", "1024"))
# tail 模式带回的字数；summary 模式压缩失败时也退回带末尾这么多字（0 = 不带回）
TRUNCATE_REASONING_TAIL = int(os.getenv("TRUNCATE_REASONING_TAIL", "4000"))
# ---- 复杂度分流：每步先用一次不思考的独立调用判断复杂度，再按档位决定思考强度与 max_tokens ----
COMPLEXITY_CHECK = os.getenv("COMPLEXITY_CHECK", "1") == "1"   # 0 = 关闭，回到 THINK_LADDER 全阶梯
# 复杂度 → [起始档位, max_tokens]；重试时从起始档位沿 THINK_LADDER 往下降
COMPLEXITY_PROFILE = json.loads(os.getenv("COMPLEXITY_PROFILE") or
                                '{"simple": ["off", 4096], "medium": ["low", 8192], "complex": ["high", 32768]}')
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
# 改用模型自己候选里不丢子的那个；候选都丢子时让模型从全部合法着法里按规则不丢子的名单中挑，
# 一个不丢子的都没有就走净亏最小的。这是对局中唯一由程序计算的检查
HANG_GUARD = os.getenv("HANG_GUARD", "1") == "1"
HANG_GUARD_MIN = int(os.getenv("HANG_GUARD_MIN", "2"))      # 默认 2：放过弃一兵的开局弃兵
HANG_GUARD_ROUNDS = int(os.getenv("HANG_GUARD_ROUNDS", "2"))
# 净亏 ≤ 该值时，模型可以不写变化、只凭局面性补偿的理由坚持（默认 2：弃半子 / 一子换兵）；0 = 一律要变化
HANG_GUARD_POSITIONAL = int(os.getenv("HANG_GUARD_POSITIONAL", "2"))
# 将杀守卫：按规则检查走完某步后对方有没有一步杀，有就只告诉模型"存在一步杀"（不给具体着法）让它改选，
# 最多 MATE_GUARD_ROUNDS 轮，仍未解决时改用候选里（再不行从合法着法里）不会被一步杀的着法。
# 同时：任一方存在一步杀时不背谱、不进开局快速模式；对方有一步杀威胁时在 prompt 里提醒一句（同样不给着法）
MATE_GUARD = os.getenv("MATE_GUARD", "1") == "1"
MATE_GUARD_ROUNDS = int(os.getenv("MATE_GUARD_ROUNDS", "2"))
# 守卫的赛后统计（Stockfish 评估每次触发时的原着法与最终着法），逐局追加
GUARD_STATS_PATH = os.getenv("GUARD_STATS_PATH", os.path.join("data", "guard_stats.jsonl"))
GUARD_OK_DELTA = int(os.getenv("GUARD_OK_DELTA", "50"))   # 着法损失 < 该值（cp）算"其实没问题"

# 模型的思考过程：LOG_REASONING=1 时完整打印到日志；每步的完整记录（思考 / 自检 / 守卫）
# 另外逐局存到 GAME_ARCHIVE_DIR/<时间>_<对局>.jsonl（live/state.json 每局会被清空）
LOG_REASONING = os.getenv("LOG_REASONING", "1") == "1"
GAME_ARCHIVE_DIR = os.getenv("GAME_ARCHIVE_DIR", os.path.join("logs", "games"))
# 自检的思考档位上限：决策档位高于它时，自检降到该档（空 = 不封顶，沿用决策档位）
SELF_CHECK_EFFORT = os.getenv("SELF_CHECK_EFFORT", "low").strip().lower()


SNAPSHOT_OK_DELTA = 50        # 走子后己方评估损失 < 该值（cp）才算"好棋"，才允许入库
SNAPSHOT_DEDUPE_SIM = 0.95    # 与已有条目余弦相似度 ≥ 该值则视为重复，跳过
# 教训类条目写入时，与已有同类教训余弦相似度 ≥ 该值就合并（seen +1），不再重复追加。
# 实测（768 维 embedding）≥0.90 基本是同一条教训换了说法，0.88 左右已是相关但不同的教训
LESSON_MERGE_SIM = float(os.getenv("LESSON_MERGE_SIM", "0.90"))
# 赛后复盘的并行线程数：逐个 blunder 的分析、快照的 Stockfish 验证各自用这么多线程（每个线程一个 Stockfish 进程）
REVIEW_WORKERS = int(os.getenv("REVIEW_WORKERS", "4"))

# ---- 背谱：记住自己走过、且局面没有变差的开局，之后遇到同样局面按概率直接照走 ----
BOOK_ENABLED = os.getenv("BOOK_ENABLED", "1") == "1"
BOOK_PATH = os.getenv("BOOK_PATH", os.path.join("data", "opening_book.json"))
BOOK_PLAY_PROB = float(os.getenv("BOOK_PLAY_PROB", "0.6"))   # 命中背谱时直接照走的基础概率（n=1），其余重新推理
# 谱里这一步的赛后评估平均 < 0 时改用 BOOK_NEG_PLAY_PROB / (n + 1)：被选得越多越少背；低于 BOOK_FLOOR_CP 的只做提醒不背
BOOK_NEG_PLAY_PROB = float(os.getenv("BOOK_NEG_PLAY_PROB", "0.2"))
# 赛后用 Stockfish 评估我方每步走完后的局面（我方视角）；第一次低于该值的那步及之后都不入谱（-100 = -1.0 兵）
BOOK_FLOOR_CP = int(os.getenv("BOOK_FLOOR_CP", "-100"))
BOOK_MAX_MOVES = int(os.getenv("BOOK_MAX_MOVES", "15"))      # 每局最多记我方前几步
