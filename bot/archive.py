"""保存下棋模型（LLM_MODEL）返回的思考过程。

- log_reasoning：把某次调用的完整思考打印到日志（LOG_REASONING=0 时只打印字数）。
- append_decision：每步的完整记录（思考 / 自检 / 守卫）逐行追加到 logs/games/<对局>.jsonl，
  不像 live/state.json 那样每局开始时被清空。
"""
import json
import os

from .config import GAME_ARCHIVE_DIR, LOG_REASONING


def log_reasoning(tag: str, text: str):
    if not text:
        return
    if LOG_REASONING:
        print(f"[REASONING {tag}] {len(text)} 字\n{text}\n[/REASONING {tag}]")
    else:
        print(f"[REASONING {tag}] {len(text)} 字")


def append_decision(game_id: str, record: dict):
    """追加一行 JSON；失败只打印，不影响下棋。"""
    if not game_id:
        return
    try:
        os.makedirs(GAME_ARCHIVE_DIR, exist_ok=True)
        path = os.path.join(GAME_ARCHIVE_DIR, f"{game_id}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ARCHIVE] failed: {e}")
