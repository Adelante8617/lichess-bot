"""技能整理流程的离线测试：假的 LLM 按提示词种类返回预设回答。

运行：python -m unittest discover -s tests
"""
import os
import shutil
import tempfile
import types
import unittest

import testenv  # noqa: F401  必须在导入 bot 之前：切临时目录、钉死开关

from bot import curate, skills  # noqa: E402
from bot.rag import RAGStore  # noqa: E402

THREATS = "---\nname: threats-first\ndescription: 先处理威胁\nwhen:\n  opp_captured: true\n---\n1. 先吃回。\n"
REWRITTEN = "---\nname: threats-first\ndescription: 先处理威胁\nwhen:\n  opp_captured: true\n---\n1. 先吃回。\n2. 逃子先查落点。\n"
NEW = '---\nname: rook-endgame\ndescription: 车残局\nwhen:\n  only: ["R", "P"]\n---\n1. 车在兵后。\n'


def reply(content: str):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))])


class CurateTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, "threats-first"))
        with open(os.path.join(self.root, "threats-first", "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(THREATS)
        self.store = RAGStore(os.path.join(tempfile.mkdtemp(), "e.jsonl"), lambda t: [1.0, 0.0])
        lessons = ["逃子前先查落点 (局面: ply3 白方走 Bg5, 引擎推荐 h3, Δ=300cp)", "车放在通路兵后面",
                   "车残局王要活跃", "车要活跃不要被动守兵", "保持警觉"]
        for i, t in enumerate(lessons):
            self.store.add(t, {"kind": "blunder" if i == 0 else None})
        self.store.add("[Blunder-Lesson] 被证伪的", {"kind": "blunder", "alt_verdict": "blunder"})
        self.store.add("[复盘] 整局总结", {})
        self.prompts = []
        self.saved = curate.complete

    def tearDown(self):
        curate.complete = self.saved
        shutil.rmtree(self.root, ignore_errors=True)

    def fake(self, assign: str, draft: str = NEW, rewrite: str = REWRITTEN):
        def complete(**kw):
            prompt = kw["messages"][0]["content"]
            self.prompts.append(prompt)
            if "逐条决定去向" in prompt:
                return reply(assign)
            if "新建一个技能文件" in prompt:
                return reply(f"```markdown\n{draft}```")
            return reply(rewrite)
        curate.complete = complete

    ASSIGN = ('{"assign": [{"i": 1, "to": "threats-first"}, {"i": 2, "to": "new:rook-endgame"}, '
              '{"i": 3, "to": "new:rook-endgame"}, {"i": 4, "to": "new:rook-endgame"}, {"i": 5, "to": "discard"}]}')

    def test_full_run(self):
        self.fake(self.ASSIGN)
        report = curate.curate(root=self.root, store=self.store)
        self.assertEqual(set(report["changed"]), {"threats-first"})
        self.assertEqual(set(report["created"]), {"rook-endgame"})
        loaded = skills.load_skills(self.root)
        self.assertIn("逃子先查落点", loaded["threats-first"]["body"])
        self.assertEqual(loaded["rook-endgame"]["when"], [{"only": ["R", "P"]}])  # 代码块围栏已去掉
        curated = {e["text"][:6]: e["meta"].get("curated") for e in self.store.entries}
        self.assertEqual(curated["逃子前先查落"], "threats-first")
        self.assertEqual(curated["保持警觉"], "discard")
        self.assertIsNone(curated["[Blund"])   # 被证伪的教训不参与整理
        self.assertIsNone(curated["[复盘] 整"])  # 整局总结不是教训
        classify_prompt = next(p for p in self.prompts if "逐条决定去向" in p)
        self.assertNotIn("局面: ply3", classify_prompt)  # 出处被去掉
        self.assertNotIn("被证伪的", classify_prompt)
        # 再跑一次：没有待整理的教训
        self.prompts.clear()
        report = curate.curate(root=self.root, store=self.store)
        self.assertEqual(report["assigned"], {})

    def test_dry_run_writes_nothing(self):
        self.fake(self.ASSIGN)
        curate.curate(root=self.root, store=self.store, dry_run=True)
        self.assertEqual(sorted(skills.load_skills(self.root)), ["threats-first"])
        self.assertTrue(all(not e["meta"].get("curated") for e in self.store.entries))

    def test_invalid_output_is_rejected(self):
        bad = REWRITTEN.replace("opp_captured", "made_up_condition")
        self.fake(self.ASSIGN, rewrite=bad, draft=NEW.replace("rook-endgame", "threats-first"))
        report = curate.curate(root=self.root, store=self.store)
        self.assertEqual(sorted(report["failed"]), ["threats-first", "threats-first"])
        with open(os.path.join(self.root, "threats-first", "SKILL.md"), encoding="utf-8") as f:
            self.assertEqual(f.read(), THREATS)  # 原文件不变
        # 改写失败的教训不标记，下次再整理；丢弃的照常标记
        curated = [e["meta"].get("curated") for e in self.store.entries]
        self.assertEqual(curated.count("discard"), 1)
        self.assertNotIn("threats-first", curated)

    def fake_with_merge(self, assigns: list[str], merge: str):
        """归类按批次依次返回 assigns；合并新名字的提示返回 merge。"""
        queue = list(assigns)

        def complete(**kw):
            prompt = kw["messages"][0]["content"]
            self.prompts.append(prompt)
            if "逐条决定去向" in prompt:
                return reply(queue.pop(0))
            if "各批分别提议的新主题名字" in prompt:
                return reply(merge)
            if "新建一个技能文件" in prompt:
                return reply(NEW)
            return reply(REWRITTEN)
        curate.complete = complete

    def test_synonym_new_names_are_merged_across_batches(self):
        # 两批各自起名，每组都不到 3 条；合并后 3 条，够起草新技能
        self.fake_with_merge(['{"assign": [{"i": 1, "to": "new:rook-endgame"}, {"i": 2, "to": "new:rook-endgame"}]}',
                              '{"assign": [{"i": 1, "to": "new:rook-activity"}, {"i": 2, "to": "discard"}]}',
                              '{"assign": []}'],
                             '{"merge": {"new:rook-activity": "new:rook-endgame"}}')
        report = curate.curate(root=self.root, store=self.store, batch_size=2)
        self.assertEqual(report["renamed"], {"new:rook-activity": "new:rook-endgame"})
        self.assertEqual(report["assigned"]["new:rook-endgame"], 3)
        self.assertEqual(set(report["created"]), {"rook-endgame"})
        merge_prompt = next(p for p in self.prompts if "各批分别提议的新主题名字" in p)
        self.assertIn("threats-first", merge_prompt)  # 现有技能也列出来，可以并进去

    def test_new_name_can_merge_into_existing_skill(self):
        self.fake_with_merge(['{"assign": [{"i": 1, "to": "new:hanging-pieces"}, {"i": 2, "to": "new:rook-endgame"}]}',
                              '{"assign": []}', '{"assign": []}'],
                             '{"merge": {"new:hanging-pieces": "threats-first", "new:rook-endgame": "made-up"}}')
        report = curate.curate(root=self.root, store=self.store, batch_size=2)
        self.assertEqual(report["renamed"], {"new:hanging-pieces": "threats-first"})  # 编出来的目标被忽略
        self.assertEqual(set(report["changed"]), {"threats-first"})

    def test_single_new_name_skips_merge_call(self):
        self.fake(self.ASSIGN)
        curate.curate(root=self.root, store=self.store)
        self.assertFalse(any("各批分别提议的新主题名字" in p for p in self.prompts))

    def test_run_parallel_keeps_order_and_survives_failures(self):
        def fn(x):
            if x == 3:
                raise RuntimeError("boom")
            return x * 10
        self.assertEqual(curate.run_parallel(fn, [1, 2, 3, 4], "测试"), [10, 20, None, 40])
        self.assertEqual(curate.run_parallel(fn, [], "测试"), [])

    def test_too_few_lessons_for_new_skill_wait(self):
        self.fake('{"assign": [{"i": 2, "to": "new:rook-endgame"}, {"i": 9, "to": "x"}]}')
        report = curate.curate(root=self.root, store=self.store)
        self.assertEqual(report["created"], {})
        self.assertTrue(all(not e["meta"].get("curated") for e in self.store.entries))


if __name__ == "__main__":
    unittest.main()
