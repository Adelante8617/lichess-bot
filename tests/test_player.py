"""离线测试：用假的 LLM 客户端驱动 get_llm_move / self_check，不访问网络、不需要 API key。

运行：python -m unittest discover -s tests
"""
import json
import os
import sys
import tempfile
import types
import unittest

# 在导入 bot 之前切到临时目录：RAG 库 / live 状态文件都写在当前目录下
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(tempfile.mkdtemp(prefix="lichess-bot-test-"))
# 钉死开关，不受本机 .env 影响（load_dotenv 不覆盖已存在的环境变量）
os.environ.update({"AUTO_RECALL_K": "0", "SELF_CHECK_ROUNDS": "2", "OPENING_FAST_MOVES": "0",
                   "COMPLEXITY_CHECK": "1", "THINK_LADDER": "default", "MATERIAL_LEAD_SKIP": "12",
                   "BOARD_RELATIONS": "0", "ANALYSIS_BOARD": "0", "PLAN_MEMORY": "1",
                   "HANG_GUARD": "1", "HANG_GUARD_MIN": "2", "HANG_GUARD_ROUNDS": "2",
                   "TRUNCATE_SALVAGE": "summary", "TRUNCATE_REASONING_TAIL": "4000"})

import chess  # noqa: E402

from bot import guard, llm, player  # noqa: E402


def _msg(content: str, finish: str = "stop"):
    message = types.SimpleNamespace(content=content, tool_calls=None, reasoning_content="",
                                    model_extra={})
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason=finish)])


class FakeClient:
    """按提示词种类返回预设回答；self_check_replies 依次用于每轮自检。"""

    def __init__(self, decision: dict, self_check_replies: list[dict], complexity="medium",
                 guard_replies: list[dict] | None = None):
        self.decision = decision
        self.self_check_replies = list(self_check_replies)
        self.guard_replies = list(guard_replies or [])
        self.complexity = complexity
        self.prompts: list[str] = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        last = kw["messages"][-1]["content"]
        self.kwargs = getattr(self, "kwargs", []) + [kw]
        self.prompts.append(last)
        if "只判断当前局面的复杂度" in last:
            return _msg(json.dumps({"complexity": self.complexity, "reason": "测试"}, ensure_ascii=False))
        if "程序做了一个简单的吃子交换模拟" in last:
            return _msg(json.dumps(self.guard_replies.pop(0), ensure_ascii=False))
        if "落子前" in last and "复查" in last:
            return _msg(json.dumps(self.self_check_replies.pop(0), ensure_ascii=False))
        return _msg(json.dumps(self.decision, ensure_ascii=False))


def keep(move):
    return {"verdict": "keep", "move": move, "reason": "安全"}


def change(move, reason="会丢子"):
    return {"verdict": "change", "move": move, "reason": reason}


class SelfCheckTest(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board()
        self.legal = list(player.legal_san_map(self.board))

    def run_check(self, first: str, replies: list[dict]):
        llm._client = FakeClient({}, replies)
        mv, records = player.self_check(self.board, [], self.board.parse_san(first), self.legal, "medium")
        return self.board.san(mv), records, llm._client

    def test_keep(self):
        san, records, _ = self.run_check("e4", [keep("e4")])
        self.assertEqual(san, "e4")
        self.assertEqual(len(records), 1)

    def test_change_then_keep(self):
        san, records, _ = self.run_check("e4", [change("d4"), keep("d4")])
        self.assertEqual(san, "d4")

    def test_no_flip_back_to_rejected_move(self):
        # 日志里的真实问题：Qxh2 → Qxd4 → Qxh2。第二轮想改回第一轮否决的着法，必须拒绝
        san, records, client = self.run_check("e4", [change("d4"), change("e4", "d4 也不好")])
        self.assertEqual(san, "d4")
        self.assertIn("已否决", records[-1].get("note", ""))
        # 第二轮的提示词里要告诉模型 e4 已被否决以及原因
        self.assertIn("e4", client.prompts[1])
        self.assertIn("会丢子", client.prompts[1])

    def test_last_round_change_is_marked_unverified(self):
        # 两轮都要求换：e4、d4 都已被否决，最后一轮换出的 Nf3 没有轮次再复查，
        # 仍然采用（比两步已知有问题的着法好），但要在记录里标明未经复查
        san, records, _ = self.run_check("e4", [change("d4"), change("Nf3")])
        self.assertEqual(san, "Nf3")
        self.assertIn("未经复查", records[-1].get("note", ""))


# 日志 20261008_110636 的 ply 21：白方走 Bg5，被 h6 黑兵直接吃掉
PLY21 = "r1bq1rk1/1p3pp1/p1nbpn1p/3p4/3P4/2NBBN2/PPP2PPP/R2QR1K1 w - - 0 11"


class MaterialRiskTest(unittest.TestCase):
    def risk(self, fen, san):
        board = chess.Board(fen)
        return guard.material_risk(board, board.parse_san(san))

    def test_piece_to_pawn_controlled_square(self):
        r = self.risk(PLY21, "Bg5")
        self.assertEqual(r["loss"], 2)  # hxg5 Nxg5：象换兵
        self.assertEqual(r["line"], ["hxg5", "Nxg5"])

    def test_safe_moves(self):
        for san in ("Re2", "Qd2", "Be2"):
            self.assertLessEqual(self.risk(PLY21, san)["loss"], 0, san)

    def test_capture_of_defended_pawn(self):
        self.assertEqual(self.risk(PLY21, "Bxh6")["loss"], 2)

    def test_favorable_capture_not_flagged(self):
        # 兵吃马、马再吃回：净赚 2，不算丢子
        board = chess.Board("4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1")
        self.assertLess(guard.material_risk(board, board.parse_san("exd5"))["loss"], 0)


class HangGuardTest(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board(PLY21)
        self.legal = list(player.legal_san_map(self.board))

    def run_guard(self, replies, candidates=("Bg5", "Re2")):
        llm._client = FakeClient({}, [], guard_replies=replies)
        mv, records = player.hang_guard(self.board, [], self.board.parse_san("Bg5"), self.legal,
                                        list(candidates))
        return self.board.san(mv), records, llm._client

    def test_change_to_safe_move(self):
        san, records, client = self.run_guard([change("Re2")])
        self.assertEqual(san, "Re2")
        self.assertIn("hxg5", client.prompts[0])  # 模拟出的交换序列交给模型

    def test_bogus_line_is_not_accepted(self):
        # 变化里 hxg5 Nxg5 后我方仍亏子：核对不通过，两轮后用候选里安全的 Re2
        bogus = {"verdict": "keep", "move": "Bg5", "kind": "tactical", "line": "hxg5 Nxg5",
                 "reason": "能拿回"}
        san, records, client = self.run_guard([bogus, bogus])
        self.assertEqual(san, "Re2")
        self.assertIn("摆了一遍", client.prompts[1])  # 第二轮把摆出的事实交回模型

    def test_sacrifice_with_valid_line_is_kept(self):
        # 希腊式弃象：Bxh7+ Kxh7 Ng5+ ... 一路将杀，变化经规则核对通过，照走
        board = chess.Board("rnbq1rk1/pppn1ppp/4p3/3pP3/1b1P4/2NB1N2/PPP2PPP/R1BQK2R w KQ - 0 7")
        line = "Bxh7+ Kxh7 Ng5+ Kg8 Qh5 Re8 Qxf7+ Kh8 Qh5+ Kg8 Qh7+ Kf8 Qh8+ Ke7 Qxg7"
        llm._client = FakeClient({}, [], guard_replies=[
            {"verdict": "keep", "move": "Bxh7+", "kind": "tactical", "line": line, "reason": "杀王"}])
        mv, records = player.hang_guard(board, [], board.parse_san("Bxh7+"),
                                        list(player.legal_san_map(board)), [])
        self.assertEqual(board.san(mv), "Bxh7+")
        self.assertEqual(records[0]["outcome"], "kept_tactical")

    def test_positional_keep_without_line(self):
        board = chess.Board("rnbq1rk1/pppn1ppp/4p3/3pP3/1b1P4/2NB1N2/PPP2PPP/R1BQK2R w KQ - 0 7")
        llm._client = FakeClient({}, [], guard_replies=[
            {"verdict": "keep", "move": "Bxh7+", "kind": "positional", "line": "", "reason": "王翼被削弱"}])
        mv, records = player.hang_guard(board, [], board.parse_san("Bxh7+"),
                                        list(player.legal_san_map(board)), [])
        self.assertEqual(board.san(mv), "Bxh7+")
        self.assertEqual(records[0]["outcome"], "kept_positional")

    def test_guard_prompt_is_neutral(self):
        _, _, client = self.run_guard([change("Re2")])
        self.assertNotIn("不会算错", client.prompts[0])
        self.assertIn("只是一条参考信息", client.prompts[0])

    def test_keep_without_line_falls_back_to_safe_candidate(self):
        san, records, _ = self.run_guard([keep("Bg5")])
        self.assertEqual(san, "Re2")
        self.assertIn("不丢子", records[-1]["note"])

    def test_change_to_another_hanging_move_is_rechecked(self):
        san, records, client = self.run_guard([change("Bf4"), change("Bg5")], candidates=["Qd2"])
        self.assertEqual(san, "Qd2")  # Bf4 也丢象，想改回 Bg5 被拒，最后用候选里安全的 Qd2
        self.assertIn("Bg5", client.prompts[1])  # 第二轮提示里列出已查出丢子的 Bg5

    def test_no_llm_call_for_safe_move(self):
        llm._client = FakeClient({}, [])
        mv, records = player.hang_guard(self.board, [], self.board.parse_san("Re2"), self.legal, [])
        self.assertEqual(records, [])
        self.assertEqual(llm._client.prompts, [])


class BookTest(unittest.TestCase):
    LINE = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]  # 白第 1、2、3 步在下标 0、2、4

    def setUp(self):
        from bot import book
        self.book = book
        book._book = {}
        book.config.BOOK_PATH = os.path.join(tempfile.mkdtemp(), "book.json")

    def tearDown(self):
        self.book._book = {}  # 别让谱漏到别的用例里

    def test_keepers_stop_at_first_disadvantage(self):
        # 第 3 步（下标 4）走完评估 -150 < -100：它和之后都不入谱，前两步保留
        evals = [(0, 20), (2, -30), (4, -150), (6, 10)]
        self.assertEqual(self.book.keepers(evals, -100), [(0, 20), (2, -30)])
        self.assertEqual(self.book.keepers([(0, -100)], -100), [(0, -100)])  # 恰好 -1.0 不算劣
        self.assertEqual(self.book.keepers([(0, -101)], -100), [])

    def test_record_and_lookup_with_transposition(self):
        self.assertEqual(self.book.record_line(self.LINE, [(0, 20), (2, -30)]), 2)
        board = chess.Board()
        self.assertEqual(self.book.lookup(board)["uci"], "e2e4")
        board.push_san("e4"), board.push_san("e5")
        self.assertEqual(self.book.lookup(board)["uci"], "g1f3")
        board.push_san("Nf3"), board.push_san("Nc6")  # 谱里到此为止（第 3 步没入谱）
        self.assertIsNone(self.book.lookup(board))

    def test_transposition_hits(self):
        # 谱里是 1.d4 Nf6 2.c4 e6 3.Nc3；用 1.c4 Nf6 2.d4 e6 换序走到同一局面也要命中
        line = ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3"]
        self.book.record_line(line, [(0, 10), (2, 10), (4, 10)])
        other = chess.Board()
        for san in ("c4", "Nf6", "d4", "e6"):
            other.push_san(san)
        self.assertEqual(self.book.lookup(other)["uci"], "b1c3")

    def test_prefers_better_average_eval_and_persists(self):
        self.book.record_line(["e2e4"], [(0, 10)])
        self.book.record_line(["d2d4"], [(0, 60)])
        self.assertEqual(self.book.lookup(chess.Board())["uci"], "d2d4")
        self.book._save()
        self.book._book = None  # 模拟重启后重新读盘
        self.assertEqual(self.book.lookup(chess.Board())["uci"], "d2d4")

    def test_play_probability(self):
        self.book.record_line(["e2e4"], [(0, 10)])
        decision = {"strategy": "x", "think": "x", "pv": ["d4"], "move": "d4"}
        client = FakeClient(decision, [keep("d4")])
        llm._client = client
        original = player.random.random
        try:
            player.random.random = lambda: 0.59  # < 0.6：直接背谱，不调用 LLM
            uci, think, _, obs = player.get_llm_move(chess.Board(), 1, None, None)
            self.assertEqual(uci, "e2e4")
            self.assertIn("背谱", think)
            self.assertEqual(client.prompts, [])
            player.random.random = lambda: 0.61  # ≥ 0.6：重新推理
            uci, *_ = player.get_llm_move(chess.Board(), 1, None, None)
            self.assertEqual(uci, "d2d4")
            self.assertTrue(client.prompts)
        finally:
            player.random.random = original

    def test_entries_remember_how_the_position_was_reached(self):
        self.book.record_line(self.LINE, [(0, 20), (2, -30)])
        board = chess.Board()
        board.push_san("e4"), board.push_san("e5")
        moves = self.book._positions()[board.epd()]
        self.assertIn("e4 e5", moves["g1f3"]["via"])

    def test_play_prob_formula(self):
        for n, want in ((1, 0.6), (2, 0.8), (3, 0.6 + 0.4 * (1 - 1 / 3)), (10, 0.96)):
            self.assertAlmostEqual(self.book.play_prob(n), want)
        self.assertLess(self.book.play_prob(1000), 1.0)
        # 评估恰好为 0 仍按原公式
        self.assertAlmostEqual(self.book.play_prob(1, 0), 0.6)

    def test_play_prob_negative_eval(self):
        # -1.0 兵 ≤ 评分 < 0：0.2/(n+1)，被选得越多越少背
        self.assertAlmostEqual(self.book.play_prob(1, -30), 0.1)
        self.assertAlmostEqual(self.book.play_prob(3, -30), 0.05)
        self.assertGreater(self.book.play_prob(1, -30), self.book.play_prob(9, -30))
        # 评分低于下限（-100cp）的是错棋记录，不背
        self.assertEqual(self.book.play_prob(1, -150), 0.0)

    def test_negative_eval_move_replays_with_small_probability(self):
        self.book.record_line(["e2e4"], [(0, -50)])  # 平均评估 -0.5：n=1 时概率 0.1
        client = FakeClient({"strategy": "x", "think": "x", "pv": ["d4"], "move": "d4"}, [keep("d4")])
        llm._client = client
        original = player.random.random
        try:
            player.random.random = lambda: 0.05
            uci, *_ = player.get_llm_move(chess.Board(), 1, None, None)
            self.assertEqual(uci, "e2e4")
            self.assertEqual(client.prompts, [])
            player.random.random = lambda: 0.15
            uci, *_ = player.get_llm_move(chess.Board(), 1, None, None)
            self.assertEqual(uci, "d2d4")
        finally:
            player.random.random = original

    def test_first_bad_move_is_recorded_as_warning_not_replayed(self):
        evals = [(0, 20), (2, -250), (4, 50)]
        bad = self.book.faulty(evals, -100)
        self.assertEqual(bad, (2, -250))
        self.book.record_line(self.LINE, self.book.keepers(evals, -100) + [bad])
        board = chess.Board()
        board.push_san("e4"), board.push_san("e5")
        self.assertIsNone(self.book.lookup(board))  # 只有错棋记录：不背
        warn = self.book.warnings(board)
        self.assertEqual([(w["san"], w["n"], round(w["avg_cp"])) for w in warn], [("Nf3", 1, -250)])

    def test_warnings_show_three_lowest(self):
        for uci, cp in (("e2e4", -20), ("d2d4", -300), ("g1f3", -90), ("c2c4", -150), ("b1c3", 40)):
            self.book.record_line([uci], [(0, cp)])
        warn = self.book.warnings(chess.Board())
        self.assertEqual([w["san"] for w in warn], ["d4", "c4", "Nf3"])  # 评分最低的三个，正分的不提醒

    def test_best_move_chosen_and_ties_are_random(self):
        for uci, cp in (("e2e4", 30), ("d2d4", 30), ("g1f3", 10)):
            self.book.record_line([uci], [(0, cp)])
        seen = {self.book.lookup(chess.Board())["uci"] for _ in range(60)}
        self.assertEqual(seen, {"e2e4", "d2d4"})  # 并列最高随机，评分更低的 Nf3 从不被选
        # 次数多不影响：e4 记过三次也不压过同分的 d4
        self.book.record_line(["e2e4"], [(0, 30)])
        self.book.record_line(["e2e4"], [(0, 30)])
        seen = {self.book.lookup(chess.Board())["uci"] for _ in range(60)}
        self.assertEqual(seen, {"e2e4", "d2d4"})

    def test_warning_is_in_decision_prompt(self):
        self.book.record_line(["e2e4"], [(0, -250)])
        self.book._positions()[chess.Board().epd()]["e2e4"]["confirm"] = 2  # 共选过 3 次
        client = FakeClient({"strategy": "x", "think": "x", "pv": ["d4"], "move": "d4"}, [keep("d4")])
        llm._client = client
        uci, _, _, obs = player.get_llm_move(chess.Board(), 1, None, None)
        self.assertEqual(uci, "d2d4")
        prompt = next(p for p in client.prompts if "【当前局面，轮到你走】" in p)
        self.assertIn("历史上在这个局面已经选择过 3 次", prompt)
        self.assertIn("-2.50", prompt)
        self.assertIn("谨慎", prompt)
        self.assertTrue(obs["book_warning"])

    def test_unseen_position_always_reasons(self):
        self.book.record_line(["e2e4"], [(0, 10)])
        board = chess.Board()
        board.push_san("c4")  # 谱外局面
        board.push_san("e5")
        client = FakeClient({"strategy": "x", "think": "x", "pv": ["Nc3"], "move": "Nc3"}, [keep("Nc3")])
        llm._client = client
        original = player.random.random
        try:
            player.random.random = lambda: 0.0
            uci, *_ = player.get_llm_move(board, 3, None, None)
        finally:
            player.random.random = original
        self.assertEqual(uci, "b1c3")


class ArchiveTest(unittest.TestCase):
    def test_decision_reasoning_is_saved_per_game(self):
        from bot.live import live
        live.start_game("archive_test", "local", "白")
        live.decision(1, move_san="e4", reasoning="测试思考内容")
        live.decision(3, move_san="Nf3", reasoning="第二步的思考")
        path = os.path.join("logs", "games", "archive_test.jsonl")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual([r["ply"] for r in rows], [1, 3])
        self.assertEqual(rows[0]["reasoning"], "测试思考内容")


class GetMoveTest(unittest.TestCase):
    def test_guard_runs_in_full_flow(self):
        board = chess.Board(PLY21)
        decision = {"strategy": "x", "candidates": [{"move": "Bg5"}, {"move": "Re2"}], "think": "x",
                    "pv": ["Bg5"], "move": "Bg5"}
        llm._client = FakeClient(decision, [keep("Re2")], guard_replies=[change("Re2")])
        uci, _, _, obs = player.get_llm_move(board, 21, None, None)
        self.assertEqual(uci, "e1e2")
        self.assertEqual(obs["guard"][0]["changed_to"], "Re2")

    def test_full_flow(self):
        board = chess.Board()
        decision = {"strategy": "抢占中心", "candidates": [{"move": "e4"}], "think": "x",
                    "pv": ["e4", "e5", "Nf3"], "board_summary": "开局", "move": "e4"}
        llm._client = FakeClient(decision, [keep("e4")])
        uci, think, _, obs = player.get_llm_move(board, 1, None, None)
        self.assertEqual(uci, "e2e4")
        self.assertEqual(obs["complexity"], "medium")

    def test_strategy_carried_to_next_move(self):
        board = chess.Board()
        decision = {"strategy": "抢占中心后王车易位", "think": "x", "pv": ["e4"], "move": "e4"}
        llm._client = FakeClient(decision, [keep("e4")])
        uci, *_ = player.get_llm_move(board, 1, None, None)
        board.push_uci(uci)
        prev = board.copy()
        board.push_san("e5")
        llm._client = FakeClient(dict(decision, move="Nf3", pv=["Nf3"]), [keep("Nf3")])
        player.get_llm_move(board, 3, prev, "e7e5")
        decision_prompt = next(p for p in llm._client.prompts if "【当前局面，轮到你走】" in p)
        self.assertIn("你上一步定下的战略方针", decision_prompt)
        self.assertIn("抢占中心后王车易位", decision_prompt)

    def test_strategy_stage_limits_candidates(self):
        board = chess.Board()
        client = FakeClient({"think": "x", "pv": ["d4"], "move": "d4"}, [keep("d4")])
        stage_reply = {"urgent": "无", "strategy": "抢中心",
                       "candidates": [{"move": "d4"}, {"move": "Ke2"}, {"move": "e4"}]}
        original = client._create

        def create(**kw):
            if "先不要选着" in kw["messages"][-1]["content"]:
                client.prompts.append(kw["messages"][-1]["content"])
                return _msg(json.dumps(stage_reply, ensure_ascii=False))
            return original(**kw)

        client.chat.completions.create = create
        llm._client = client
        player.STRATEGY_STAGE = True
        try:
            uci, _, _, obs = player.get_llm_move(board, 1, None, None)
        finally:
            player.STRATEGY_STAGE = False
        self.assertEqual(uci, "d2d4")
        self.assertEqual(obs["strategy"], "抢中心")  # 决策回答没给 strategy 时沿用第一阶段
        decision_prompt = next(p for p in client.prompts if "【当前局面，轮到你走】" in p)
        self.assertIn("第一阶段已定下的结论", decision_prompt)
        self.assertNotIn("- Ke2", decision_prompt)  # 非法候选被过滤

    def test_opening_fast_mode_offers_no_tools_and_skips_recall(self):
        decision = {"strategy": "抢中心", "think": "x", "pv": ["e4"], "move": "e4"}
        client = FakeClient(decision, [])
        llm._client = client
        player.OPENING_FAST_MOVES = 8
        calls = []
        original = player.recall_experience
        player.recall_experience = lambda b: calls.append(1) or []
        try:
            uci, *_ = player.get_llm_move(chess.Board(), 1, None, None)
        finally:
            player.OPENING_FAST_MOVES = 0
            player.recall_experience = original
        self.assertEqual(uci, "e2e4")
        self.assertEqual(calls, [])
        self.assertTrue(all("tools" not in kw for kw in client.kwargs))

    def test_illegal_then_fallback(self):
        board = chess.Board()
        llm._client = FakeClient({"move": "Ke2"}, [])
        uci, *_ = player.get_llm_move(board, 1, None, None)
        self.assertIn(chess.Move.from_uci(uci), board.legal_moves)


class TruncateSalvageTest(unittest.TestCase):
    """思考被截断（finish_reason=length、正文为空）后的降档补救。"""

    REASONING = "开头分析 e4 和 d4，Nfxd4 会被 cxd4 吃回……" + "x" * 5000 + "末尾：倾向 d4"

    def make_client(self, summary_reply):
        calls: list[dict] = []

        def create(**kw):
            calls.append(kw)
            if "被截断的思考" in kw["messages"][-1]["content"]:
                if isinstance(summary_reply, Exception):
                    raise summary_reply
                return _msg(summary_reply)
            if len(calls) == 1:  # 第一档：思考用光 max_tokens，正文为空
                resp = _msg("", finish="length")
                resp.choices[0].message.reasoning_content = self.REASONING
                return resp
            return _msg(json.dumps({"move": "d4"}))

        llm._client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        return calls

    def test_summary_passed_to_next_level(self):
        calls = self.make_client("1. 候选：e4 可行，d4 可行\n- Qxh7+：可行\n3. 倾向 d4")
        msg, level = llm.llm_call([{"role": "user", "content": "走一步"}], levels=["high", "low"])
        self.assertEqual(level, "low")
        self.assertEqual(len(calls), 3)  # high 截断 → 要点整理 → low
        self.assertEqual(calls[1]["extra_body"]["thinking"], {"type": "disabled"})
        self.assertIn(self.REASONING, calls[1]["messages"][-1]["content"])  # 整理用的是完整思考
        retry = calls[2]["messages"][-1]["content"]
        self.assertIn("要点整理", retry)
        self.assertIn("倾向 d4", retry)
        self.assertIn("可能有错", retry)
        self.assertIn("对照上面的棋盘核实", retry)
        self.assertNotIn("x" * 100, retry)  # 不再贴原始末尾
        self.assertNotIn("Qxh7", retry)  # 原文没出现过的着法不给下棋模型
        self.assertIn("[被截断思考的要点整理]", msg.reasoning_content)
        self.assertIn("Qxh7", msg.reasoning_content)  # 但日志里看得到被删的行

    def test_drop_unseen_moves(self):
        summary = ("1. 候选\n- Nxd4：被 cxd4 吃回\n- Bb5+：未算完\n- d8=Q 升变\n- O-O：可行\n"
                   "2. 威胁：c6 格被控制，g5 兵步不检查\n3. 倾向 Nxd4")
        kept, dropped = llm.drop_unseen_moves(summary, "Nfxd4? cxd4. 也想过 d8=Q+ 和 O-O")
        self.assertIn("Nxd4：被 cxd4 吃回", kept)  # 忽略消歧字母：Nxd4 = Nfxd4
        self.assertIn("d8=Q", kept)
        self.assertIn("O-O", kept)
        self.assertIn("c6 格", kept)  # 格子名、兵步不当作着法
        self.assertIn("倾向 Nxd4", kept)
        self.assertEqual(len(dropped), 1)
        self.assertIn("Bb5", dropped[0])

    def test_summary_failure_falls_back_to_tail(self):
        calls = self.make_client(RuntimeError("boom"))
        _, level = llm.llm_call([{"role": "user", "content": "走一步"}], levels=["high", "low"])
        self.assertEqual(level, "low")
        retry = calls[-1]["messages"][-1]["content"]
        self.assertIn("思考的最后部分", retry)
        self.assertIn("末尾：倾向 d4", retry)
        self.assertNotIn("对照上面的棋盘核实", retry)  # 带末尾时仍是原来的直接要结论


if __name__ == "__main__":
    unittest.main()
