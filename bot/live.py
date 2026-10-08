"""
实时对局状态：游戏进程把当前局面 / 思考过程写到 live/state.json，
viewer.py 读取它并在浏览器里展示。所有方法都吞掉异常，不会影响下棋本身。
"""
import functools
import json
import os
import threading
import time

import chess

from .archive import append_decision

LIVE_PATH = os.getenv("LIVE_STATE_PATH", os.path.join("live", "state.json"))


def _safe(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        try:
            with self.lock:
                fn(self, *a, **kw)
                self._write()
        except Exception as e:  # 展示层出问题绝不能拖垮下棋
            print(f"[LIVE] {fn.__name__} failed: {e}")
    return wrapper


class LiveState:
    def __init__(self, path: str = LIVE_PATH):
        self.path = path
        self.lock = threading.Lock()
        self.s = self._blank()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    @staticmethod
    def _blank() -> dict:
        return {
            "game_id": "", "mode": "", "my_color": "", "opponent": "",
            "status": "idle",        # idle / thinking / waiting_opponent / reviewing / finished
            "result": "", "ply": 0,
            "moves_san": [], "moves_uci": [], "fens": [chess.STARTING_FEN],
            "thinking_ply": None, "thinking_since": None, "current_tools": [], "stage": "",
            "decisions": {},         # str(ply) -> 我方那一步的思考 / 观察 / 工具调用
            "updated_at": time.time(),
        }

    def _write(self):
        self.s["updated_at"] = time.time()
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.s, f, ensure_ascii=False)
        for _ in range(5):  # Windows 上读者正占用文件时 replace 可能失败，重试
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                time.sleep(0.02)

    # ---------- 对外接口 ----------
    @_safe
    def start_game(self, game_id: str, mode: str, my_color: str, opponent: str = ""):
        self.s = self._blank()
        self.s.update(game_id=game_id, mode=mode, my_color=my_color,
                      opponent=opponent, status="waiting_opponent")

    @_safe
    def sync(self, moves_uci: list):
        """用完整着法序列重建局面（重连 / 任意一方走子后都可直接调用）。"""
        board = chess.Board()
        sans, fens = [], [board.fen()]
        for u in moves_uci:
            mv = chess.Move.from_uci(u)
            sans.append(board.san(mv))
            board.push(mv)
            fens.append(board.fen())
        self.s.update(moves_uci=list(moves_uci), moves_san=sans, fens=fens, ply=len(moves_uci))

    @_safe
    def thinking(self, ply: int):
        self.s.update(status="thinking", thinking_ply=ply,
                      thinking_since=time.time(), current_tools=[], stage="思考中")

    @_safe
    def stage(self, text: str):
        """思考阶段的文字提示，如"第 1 轮自检：复查 Nf3"。"""
        self.s["stage"] = text

    @_safe
    def tool(self, name: str, args: dict, result: str):
        self.s["current_tools"].append(
            {"name": name, "args": args, "result": str(result)[:800]})

    @_safe
    def decision(self, ply: int, **fields):
        since = self.s.get("thinking_since")
        self.s["decisions"][str(ply)] = dict(
            fields, ply=ply, tools=list(self.s["current_tools"]),
            elapsed=round(time.time() - since, 1) if since else None)
        append_decision(self.s["game_id"], self.s["decisions"][str(ply)])
        self.s.update(status="waiting_opponent", thinking_since=None, stage="")

    @_safe
    def set_status(self, status: str, result: str = ""):
        self.s["status"] = status
        if result:
            self.s["result"] = result


live = LiveState()
