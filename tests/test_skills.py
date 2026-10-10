"""技能系统的离线测试：解析、触发条件、匹配排序、提示词注入、load_skill 工具、赛后统计。

运行：python -m unittest discover -s tests
"""
import json
import os
import tempfile
import unittest

import testenv

import chess  # noqa: E402

from bot import prompts, skills, tools  # noqa: E402


def write_skill(root: str, name: str, text: str):
    os.makedirs(os.path.join(root, name), exist_ok=True)
    with open(os.path.join(root, name, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(text)


def board_after(*sans: str, fen: str | None = None) -> chess.Board:
    board = chess.Board(fen) if fen else chess.Board()
    for san in sans:
        board.push_san(san)
    return board


class ParseTest(unittest.TestCase):
    def test_single_group_and_json_values(self):
        s = skills.parse_skill('---\nname: a\ndescription: d\npriority: 2\nwhen:\n  phase: ["中局", "残局"]\n'
                               '  lead_min: 3\n---\n正文\n第二行\n')
        self.assertEqual(s["when"], [{"phase": ["中局", "残局"], "lead_min": 3}])
        self.assertEqual(s["priority"], 2)
        self.assertEqual(s["body"], "正文\n第二行")

    def test_any_of_groups(self):
        s = skills.parse_skill("---\nname: a\ndescription: d\nwhen:\n  - opp_captured: true\n"
                               "  - threatened: true\n    in_check: false\n---\nx")
        self.assertEqual(s["when"], [{"opp_captured": True}, {"threatened": True, "in_check": False}])

    def test_inline_when_and_no_when(self):
        self.assertEqual(skills.parse_skill('---\nname: a\ndescription: d\nwhen: {"queens": false}\n---\nx')["when"],
                         [{"queens": False}])
        self.assertEqual(skills.parse_skill("---\nname: a\ndescription: d\n---\nx")["when"], [])

    def test_errors(self):
        for bad in ("name: a\n", "---\nname: a\n---\nx", "---\nname: a\ndescription: d\nwhen:\n  bogus: 1\n---\n"):
            with self.assertRaises(ValueError):
                skills.parse_skill(bad)

    def test_shipped_skills_are_valid(self):
        loaded = skills.load_skills(testenv.SKILLS_ROOT)
        dirs = [d for d in os.listdir(testenv.SKILLS_ROOT) if os.path.isdir(os.path.join(testenv.SKILLS_ROOT, d))]
        self.assertEqual(sorted(loaded), sorted(dirs))  # 每个目录都解析成功，且 name 与目录名一致
        for s in loaded.values():
            self.assertLessEqual(len(s["body"]), 1200, s["name"])


class PredicateTest(unittest.TestCase):
    def check(self, key, value, board):
        return skills.PREDICATES[key](board, value)

    def test_king_spot(self):
        board = board_after("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O")
        self.assertEqual(skills.king_spot(board, chess.WHITE), "kingside")
        self.assertEqual(skills.king_spot(board, chess.BLACK), "uncastled")
        self.assertTrue(self.check("opp_king", "kingside", board))  # 轮到黑走，对方 = 白
        self.assertTrue(self.check("my_king", "uncastled", board))

    def test_opp_captured_and_threatened(self):
        board = board_after("e4", "d5", "exd5")
        self.assertTrue(self.check("opp_captured", True, board))
        self.assertFalse(self.check("threatened", True, board))
        board = board_after("e4", "e5", "Nf3", "Nc6", "d4", "a6", "d5")  # 兵攻马
        self.assertTrue(self.check("threatened", True, board))

    def test_passer_rank(self):
        board = chess.Board("8/2P5/8/8/8/8/5k2/K7 b - - 0 1")
        self.assertEqual(skills.passer_rank(board, chess.WHITE), 7)
        self.assertTrue(self.check("opp_passer_min", 6, board))
        blocked = chess.Board("8/2P5/1p6/8/8/8/5k2/K7 b - - 0 1")  # b6 黑兵不在 c7 前方
        self.assertEqual(skills.passer_rank(blocked, chess.WHITE), 7)
        blocked = chess.Board("2p5/2P5/8/8/8/8/5k2/K7 b - - 0 1")
        self.assertEqual(skills.passer_rank(blocked, chess.WHITE), 0)

    def test_only_has_queens(self):
        board = chess.Board("8/5k2/8/3pP3/8/8/8/4K3 w - - 0 40")
        self.assertTrue(self.check("only", ["P"], board))
        self.assertFalse(self.check("has", ["R"], board))
        self.assertTrue(self.check("queens", False, board))

    def test_opening_prefix_without_move_numbers(self):
        board = board_after("e4", "e6", "d4", "d5")
        self.assertEqual(skills.san_moves(board), "e4 e6 d4 d5")
        self.assertTrue(self.check("opening", ["e4 e6"], board))
        self.assertFalse(self.check("opening", ["e4 e5"], board))
        self.assertFalse(self.check("opening", ["e4 e"], board))  # 按整步匹配


class MatchTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        write_skill(self.root, "endgame", "---\nname: endgame\ndescription: 残局\nwhen:\n  phase: 残局\n---\nE")
        write_skill(self.root, "pawns", "---\nname: pawns\ndescription: 兵残局\npriority: 3\nwhen:\n"
                                        '  only: ["P"]\n---\nP')
        write_skill(self.root, "rooks", '---\nname: rooks\ndescription: 车残局\nwhen:\n  phase: 残局\n  has: ["R"]\n'
                                        "---\nR")
        write_skill(self.root, "manual", "---\nname: manual\ndescription: 只能手动读\n---\nM")
        self.skills = skills.load_skills(self.root)

    def test_order_and_limit(self):
        pawn_ending = chess.Board("8/5k2/8/3pP3/8/8/8/4K3 w - - 0 40")
        names = [s["name"] for s in skills.match_skills(pawn_ending, self.skills, limit=5)]
        self.assertEqual(names, ["pawns", "endgame"])  # priority 高的在前；manual 没有 when 不自动加载
        rook_ending = chess.Board("8/5k2/8/3pP3/8/8/8/R3K3 w - - 0 40")
        names = [s["name"] for s in skills.match_skills(rook_ending, self.skills, limit=1)]
        self.assertEqual(names, ["rooks"])  # 条件更多（更具体）的在前

    def test_prompt_and_tool(self):
        hits = skills.match_skills(chess.Board("8/5k2/8/3pP3/8/8/8/4K3 w - - 0 40"), self.skills, limit=1)
        text = prompts.user_prompt(board_text="", last_move="", history="", my_color="白", my_pieces="",
                                   opp_pieces="", meta="", ply=1, relations="", prev_strategy=None, recalled="",
                                   legal_sans=["e6"], fast=False, complexity="", complexity_reason="",
                                   skill_text=skills.section_text(hits))
        self.assertIn("按局面自动加载的技能", text)
        self.assertIn("【pawns】兵残局\nP", text)
        self.assertNotIn("经验库自动召回", text)  # EXPERIENCE_IN_PLAY=0
        original = skills._cache
        skills._cache = self.skills
        try:
            self.assertIn("manual：只能手动读", prompts.system_prompt())
            names = [t["function"]["name"] for t in tools.play_tools()]
            self.assertIn("load_skill", names)
            self.assertNotIn("search_experience", names)
            self.assertEqual(tools.run_tool("load_skill", {"name": "manual"}), "【manual】只能手动读\nM")
            self.assertIn("可用的技能", tools.run_tool("load_skill", {"name": "nope"}))
        finally:
            skills._cache = original


class StatsTest(unittest.TestCase):
    def test_record_game(self):
        root = tempfile.mkdtemp()
        write_skill(root, "open", "---\nname: open\ndescription: 开局\nwhen:\n  max_fullmove: 2\n---\nO")
        original = skills._cache
        skills._cache = skills.load_skills(root)
        path = os.path.join(root, "stats.json")
        try:
            pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 *"
            skills.record_game(pgn, "白", [{"ply": 3, "side": "白"}, {"ply": 4, "side": "黑"}], path=path)
            skills.record_game(pgn, "黑", [], path=path)
        finally:
            skills._cache = original
        with open(path, encoding="utf-8") as f:
            stats = json.load(f)
        # 白方第 1、2 回合（ply 1、3）命中，ply 3 是白方的 blunder；黑方 ply 2、4 命中
        self.assertEqual(stats["open"], {"games": 2, "moves": 4, "blunders": 1})


if __name__ == "__main__":
    unittest.main()
