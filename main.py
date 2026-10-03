"""程序入口：python main.py 连接 Lichess 开始下棋。

实际代码在 bot/ 包里（见 README「目录结构」）。这里先初始化日志，再导出
local_play.py / laya_play.py / reembed.py 用到的名字，保持 `import main as bot` 的旧用法可用。
"""
from bot.logger_setup import setup_logger

logger, LOG_PATH = setup_logger()
print(f"=== Lichess LLM Bot started, log file: {LOG_PATH} ===")

from bot.boardtext import build_pgn  # noqa: E402
from bot.config import STOCKFISH_PATH  # noqa: E402
from bot.memory import embed, experience_rag, opening_rag, seed_openings_if_empty  # noqa: E402
from bot.player import get_llm_move  # noqa: E402
from bot.review import (blunder_deep_review, chat_review, commit_verified_snapshots,  # noqa: E402
                        post_game_review, record_snapshot)

seed_openings_if_empty()
print(f"[RAG] openings={len(opening_rag)}  experience={len(experience_rag)}")

__all__ = ["build_pgn", "STOCKFISH_PATH", "embed", "get_llm_move", "blunder_deep_review",
           "chat_review", "commit_verified_snapshots", "post_game_review", "record_snapshot"]

if __name__ == "__main__":
    from bot.lichess import run_lichess
    run_lichess()
