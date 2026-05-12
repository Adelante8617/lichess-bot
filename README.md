# Lichess LLM Chess Bot

一个挂在 [Lichess](https://lichess.org) BOT 账号下、**完全由大语言模型决策**的国际象棋机器人。
对局中不调用任何象棋引擎（Stockfish 只在赛后复盘时出现），让 LLM 自己看盘、算变、下棋；
赛后用 Stockfish 找出 blunder，逼模型自我反思，把经验写进 RAG 记忆库，下一局自动召回。
支持从 Lichess 聊天框读取人类教练的实时评价（仅赛后总结，不用于实时作弊）。

---

## 功能一览

- **LLM 直接下棋**：每一步由 `gpt-4.1`（或任意 OpenAI 兼容接口）决策，**对局阶段禁用 Stockfish**。
- **结构化思考协议**：模型输出固定 JSON，包括局面观察 / 候选着法 / 主变 PV / board_summary / think 等字段。
- **棋盘多视图 prompt**：同时给 FEN + ASCII 带坐标盘面 + 双方按子种分组的子力清单，明确标注"我方/对方"，降低"走错颜色"的概率。
- **双 FEN 对比**：每步同时给"对方走子前"与"当前"两张盘面，让模型判断对方真实意图。
- **think 预算分段**：开局 120 字 / 中局 300 字 / 残局 500 字，给中残局留出算变空间。
- **RAG 双库**：
  - `data/openings.jsonl` — 开局库，启动时播种经典开局原则。
  - `data/experience.jsonl` — 经验库，滚动增长，涵盖：实时局面快照、赛后自我反思、Stockfish 找到的 blunder 教训、聊天教练总结等。
- **工具调用分层**：对局阶段只暴露 `search_opening_book` / `search_experience`；复盘阶段额外暴露 `analyze_with_stockfish`，从制度上隔离"对局期间无外部引擎"。
- **Stockfish 强制复盘**：赛后扫描整盘 PGN 找所有 ≥200cp 的 blunder，每个让模型看两张 FEN 自行推理失误原因，提出替代走法，再用 Stockfish 评估替代走法的好坏，全部写入经验库。
- **聊天读取（只读）**：游戏中 `chatLine` 事件实时入队，附带当时 FEN/ply/last_move，**对局中模型看不到**，赛后 `chat_review` 统一总结成 `[Chat-Lesson]` 写入经验库。
- **3 次非法走法重试**：模型给出非法 UCI 时把非法原因 + 合法列表回喂，重新生成；三次失败才 fallback。
- **带时间戳的日志**：`logs/YYYYMMDD_HHMMSS.log`，`stdout/stderr` 全部重定向，便于事后排查。
- **60 秒空闲退出**：没有新挑战时优雅退出，便于通过任务计划 / cron 重启。

---

## 架构

```
Lichess 事件流 (berserk)
        │
        ▼
  主循环 (main.py)
    ├── gameFull / gameState ─► 生成 board → 轮到我方时
    │                              └─ get_llm_move(board, prev_board, opp_last_move)
    │                                    ├─ PLAY_TOOLS: search_experience / search_opening_book
    │                                    ├─ 3 次非法重试
    │                                    └─ 落子成功后把 board_summary 写入 experience_rag
    │
    ├── chatLine ─────────────► 写入 chat_messages（附当时 FEN/ply/last_move）
    │
    └── 对局结束 ─────────────► post_game_review   (自我反思 + 可选 Stockfish)
                                  blunder_deep_review (强制 Stockfish 扫 blunder)
                                  chat_review        (聊天总结)
                                        ▼
                                experience_rag (RAG, numpy 余弦)
                                        ▲
                                        │（下一局通过 search_experience 召回）
```

### 经验库：什么时候写 & 什么时候读

| 触发点 | 前缀 | meta.kind | 时机 |
|---|---|---|---|
| 落子成功 | `[InGame-Snapshot]` | `in_game_snapshot` | 对局中每步 |
| 赛后自我反思 | 教训原文 / `[复盘] ...` | — | 对局结束 |
| Stockfish blunder | `[Blunder-Lesson]` | `blunder` | 对局结束 |
| 模型替代走法评估 | `[Blunder-AltMove]` | `blunder_alt_eval` | 对局结束 |
| 聊天总结 | `[Chat-Summary]` / `[Chat-Lesson]` / `[Chat-Raw]` | `chat_guidance` / `chat_raw` | 对局结束 |

读取：仅通过 LLM 主动调用工具 `search_experience(query)`，向量 top-3 召回。不做自动注入。

---

## 快速开始

### 1. 准备账号

- 注册一个 Lichess 账号（**不能** 已经下过棋，Lichess 规定）。
- 运行 `upgrade-to-bot.py` 或访问 [账号升级](https://lichess.org/account/oauth/token/create) 把账号转为 BOT。
- 创建 token，勾选至少 `bot:play`、`challenge:read`、`challenge:write`。

### 2. 准备 Stockfish

- 从 [stockfishchess.org](https://stockfishchess.org/download/) 下载对应平台二进制。
- 放到 `PATH`，或通过 `STOCKFISH_PATH` 环境变量指定绝对路径。

### 3. 安装依赖

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env`，填入真实 key：

```env
LICHESS_TOKEN=lip_xxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxx
STOCKFISH_PATH=stockfish
STOCKFISH_DEPTH=14
WAIT_TIMEOUT_SEC=60
```

> 代码里默认用 `https://api.qingyuntop.top/v1` 作为 `OpenAI` `base_url`。如你用官方 API，改 `main.py` 里的 `OpenAI(..., base_url=...)` 那行即可。

### 5. 启动

```powershell
python main.py
```

终端会同时写一份 `logs/YYYYMMDD_HHMMSS.log`。
到 [Lichess 对局大厅](https://lichess.org/?any#hook) 用另一个账号挑战你的 BOT 即可开下。

---

## 配置项

所有配置通过环境变量（`.env`）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `LICHESS_TOKEN` | — | **必填**。Lichess BOT token |
| `OPENAI_API_KEY` | — | **必填**。LLM key |
| `STOCKFISH_PATH` | `stockfish` | Stockfish 路径 |
| `STOCKFISH_DEPTH` | `14` | 赛后分析深度，越大越慢 |
| `WAIT_TIMEOUT_SEC` | `60` | 空闲等待挑战的最大秒数 |

在 `main.py` 顶部可改：

- `MODEL`：默认 `"gpt-4.1"`
- `EMBED_MODEL`：默认 `"text-embedding-3-small"`
- LLM `base_url`：填写你的Base URL

---

## Prompt 设计要点

### SYSTEM_PROMPT

严格 JSON 输出，模型每步必须给全下列字段：

- `opp_intent` · `my_attacked` · `opp_attacked` · `my_hanging` · `opp_hanging`
- `check_chance` · `capture_chance` · `threats` · `tactics`
- `candidates[]`：3-5 个候选，每项含 `move/pros/cons`
- `pv[]`：≥3 步的主变 UCI 列表，`move` 必须 `== pv[0]`
- `board_summary`：≤80 字局面骨架（用于以后向量检索）
- `think`：综合分析，字数上限见 user_prompt 的动态预算
- `move`：最终 UCI，必须在合法走法列表里

Prompt 顶部显式声明「只能走自己颜色的子，盘面 W* = 白，B* = 黑」。

### user_prompt

每步注入：
- 对方走子前 FEN + ASCII 盘面（如果有）
- 当前 FEN + ASCII 盘面
- 我方子力清单（按 K/Q/R/B/N/P 分组）
- 对方子力清单
- 完整合法走法 UCI 列表
- 当前 think 字数上限

---

## 已知限制 / 后续可改进

- LLM 在**中残局精确计算**上弱，开局靠模式识别，越到残局越容易丢子。
- 目前每一步都开新 messages，没有历史走法上下文（这是主动选择：类比人类看盘，但会丢失自己的计划连续性）。
- 单次采样，没有 best-of-N 投票，中残局一次直觉错就落子。
- 经验库无去重 / 淘汰策略，长期运行后向量检索噪音会变大。
- 非法走法 fallback 是 `legal_moves[0]`，中残局往往导致送分——建议改进。

---

## 目录结构

```
lichess-bot/
├── main.py                 # 主程序：事件循环、LLM 决策、复盘
├── rag.py                  # 极简 RAG（numpy 余弦）
├── logger_setup.py         # 日志 + stdout/stderr 重定向
├── upgrade-to-bot.py       # 把普通账号升级为 BOT
├── requirements.txt
├── .env.example
├── LICENSE                 # MIT
├── data/                   # 运行时生成（gitignore）
│   ├── openings.jsonl
│   └── experience.jsonl
├── games/                  # 每盘 PGN 存档（gitignore）
├── logs/                   # 运行日志（gitignore）
└── stockfish/              # 自带 Stockfish 二进制（gitignore）
```

---

## License

[`MIT LICENSE`](./LICENSE)。
