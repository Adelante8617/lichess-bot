"""经验库整理的离线测试：用固定向量代替 embedding，不访问网络。

运行：python -m unittest discover -s tests
"""
import json
import os
import tempfile
import unittest

import testenv  # noqa: F401  必须在导入 bot 之前：切临时目录、钉死开关

from bot import memory  # noqa: E402
from bot.rag import RAGStore  # noqa: E402

# 文本 → 向量：同一主题的教训方向几乎相同，不同主题正交
VEC = {"象别进兵控格": [1, 0, 0], "象不要走到兵吃的格": [0.99, 0.1, 0], "先算将杀": [0, 1, 0],
       "王翼弱先防杀": [0, 0, 1]}


def fake_embed(text: str):
    return next(v for k, v in VEC.items() if k in text)


def store_at(name: str) -> RAGStore:
    return RAGStore(os.path.join(tempfile.mkdtemp(), name), fake_embed)


def lines(store: RAGStore) -> list[dict]:
    return [json.loads(x) for x in open(store.path, encoding="utf-8")]


class UpsertTest(unittest.TestCase):
    def test_similar_lesson_is_merged(self):
        s = store_at("e.jsonl")
        self.assertEqual(s.upsert("象别进兵控格", {"time": "t1"}, 0.9), "added")
        self.assertEqual(s.upsert("象不要走到兵吃的格", {"time": "t2"}, 0.9), "merged")
        self.assertEqual(s.upsert("先算将杀", {"time": "t3"}, 0.9), "added")
        rows = lines(s)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["meta"]["seen"], 2)
        self.assertEqual(rows[0]["meta"]["last_seen"], "t2")
        self.assertEqual(rows[0]["text"], "象别进兵控格")  # 保留原文
        self.assertEqual(rows[0]["meta"]["variants"], ["象不要走到兵吃的格"])  # 并入的说法不丢

    def test_filter_limits_merge_to_same_kind(self):
        s = store_at("e.jsonl")
        s.upsert("象别进兵控格", {"kind": "blunder"}, 0.9)
        self.assertEqual(s.upsert("象不要走到兵吃的格", {"kind": None}, 0.9,
                                  filter_fn=lambda e: e["meta"].get("kind") is None), "added")

    def test_remove_rewrites_file(self):
        s = store_at("e.jsonl")
        s.add("象别进兵控格")
        s.add("先算将杀")
        self.assertEqual(s.remove(lambda e: "将杀" in e["text"]), 1)
        self.assertEqual([r["text"] for r in lines(s)], ["象别进兵控格"])


class ConsolidateTest(unittest.TestCase):
    def test_refuted_and_duplicates(self):
        s = store_at("e.jsonl")
        s.add("[Blunder-Lesson] 先算将杀", {"kind": "blunder", "time": "g1", "ply": 9})
        # 旧格式：替代着法的结论只写在文本里
        s.add("[Blunder-AltMove] 王翼弱先防杀 引擎评估: cp 0->-250 (Δ=250, mistake);",
              {"kind": "blunder_alt_eval", "time": "g1", "ply": 9})
        s.add("[Blunder-Lesson] 象别进兵控格", {"kind": "blunder", "time": "g2", "ply": 3})
        s.add("[Blunder-Lesson] 象不要走到兵吃的格", {"kind": "blunder", "time": "g3", "ply": 5})
        s.add("[复盘] 象别进兵控格", {"time": "g3"})  # 整局总结不是教训，不参与合并

        report = memory.consolidate(s, 0.9, dry_run=True)
        self.assertEqual(len(report["refuted"]), 1)
        self.assertEqual(len(report["merged"]), 1)
        self.assertEqual(len(lines(s)), 5)  # dry-run 不改文件

        memory.consolidate(s, 0.9)
        texts = [r["text"] for r in lines(s)]
        self.assertNotIn("[Blunder-Lesson] 先算将杀", texts)
        self.assertNotIn("[Blunder-Lesson] 象不要走到兵吃的格", texts)
        self.assertIn("[复盘] 象别进兵控格", texts)
        kept = next(r for r in lines(s) if r["text"] == "[Blunder-Lesson] 象别进兵控格")
        self.assertEqual(kept["meta"]["seen"], 2)
        self.assertEqual(kept["meta"]["last_seen"], "g3")
        self.assertEqual(kept["meta"]["variants"], ["[Blunder-Lesson] 象不要走到兵吃的格"])


if __name__ == "__main__":
    unittest.main()
