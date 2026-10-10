"""赛后复盘的离线测试：Stockfish、LLM、经验库都换成假的，只测并行调度与写库顺序。

运行：python -m unittest discover -s tests
"""
import json
import threading
import time
import types
import unittest

import testenv  # noqa: F401  必须在导入 bot 之前：切临时目录、钉死开关

import chess  # noqa: E402

from bot import review  # noqa: E402


class FakeStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.added: list[tuple[str, dict]] = []

    def add(self, text, meta=None, **kw):
        with self.lock:
            self.added.append((text, meta or {}))
        return True

    def __len__(self):
        return len(self.added)


def fake_blunder(ply: int, san: str) -> dict:
    board = chess.Board()
    return {"ply": ply, "side": "白", "move": "", "san": san, "best": "", "best_san": "d4", "history": "",
            "board_before": "", "board_after": "", "cp_before": 0, "cp_after": -300, "delta": 300,
            "fen_before": board.fen(), "fen_after": board.fen()}


class BlunderReviewTest(unittest.TestCase):
    def setUp(self):
        self.saved = {k: getattr(review, k) for k in
                      ("stockfish_collect_blunders", "stockfish_eval_move", "complete", "experience_rag",
                       "add_lesson")}
        self.store = FakeStore()
        review.experience_rag = self.store
        review.add_lesson = lambda text, meta: self.store.add(text, meta) and "added"

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(review, k, v)

    def test_parallel_analysis_writes_in_blunder_order(self):
        blunders = [fake_blunder(p, s) for p, s in ((3, "a3"), (5, "h3"), (7, "a4"))]
        review.stockfish_collect_blunders = lambda pgn, threshold_cp: blunders
        review.stockfish_eval_move = lambda fen, uci: {"cp_before": 0, "cp_after": 0, "delta": 0,
                                                       "best": "e2e4", "verdict": "ok"}
        running, peak = [0], [0]
        lock = threading.Lock()

        def complete(**kw):
            prompt = kw["messages"][0]["content"]
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.2 if "实际走法: a3" in prompt else 0.05)  # 第一个最慢，完成顺序被打乱
            with lock:
                running[0] -= 1
            san = next(s for s in ("a3", "h3", "a4") if f"实际走法: {s}" in prompt)
            content = json.dumps({"your_better_move": "e4", "lesson": f"教训-{san}"}, ensure_ascii=False)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content=content))])

        review.complete = complete
        out = review.blunder_deep_review("pgn", "1-0", "白")
        self.assertEqual(len(out), 3)
        self.assertGreater(peak[0], 1)  # 确实并行了
        lessons = [t for t, m in self.store.added if t.startswith("[Blunder-Lesson]")]
        self.assertEqual([t.split()[1] for t in lessons], ["教训-a3", "教训-h3", "教训-a4"])

    def test_run_post_game_isolates_failures(self):
        done = []
        saved = {k: getattr(review, k) for k in ("post_game_review", "blunder_deep_review",
                                                 "commit_verified_snapshots", "commit_opening_book")}

        def boom(*a, **kw):
            raise RuntimeError("stockfish missing")

        review.post_game_review = lambda *a: done.append("review")
        review.blunder_deep_review = boom
        review.commit_verified_snapshots = lambda *a: done.append("snap")
        review.commit_opening_book = lambda *a: done.append("book")
        try:
            review.run_post_game("pgn", "1-0", "白", [], [], [], True)
        finally:
            for k, v in saved.items():
                setattr(review, k, v)
        self.assertEqual(sorted(done), ["book", "review", "snap"])


if __name__ == "__main__":
    unittest.main()
