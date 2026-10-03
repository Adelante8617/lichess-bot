"""模型可调用的工具：对局阶段只能查库 / 摆棋，复盘阶段才额外开放 Stockfish。"""
from . import board_view
from .config import ANALYSIS_BOARD
from .engine import stockfish_analyze_pgn
from .memory import experience_rag, opening_rag, search

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


def run_tool(name: str, args: dict, ctx: dict | None = None) -> str:
    ctx = ctx or {}
    q = args.get("query", "")
    if name == "search_opening_book":
        return search(opening_rag, q)
    if name == "search_experience":
        return search(experience_rag, q)
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
