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

from bot import guard, hooks, llm, player  # noqa: E402


def _delta(content=None, reasoning=None, tool_calls=None):
    return types.SimpleNamespace(role="assistant", content=content, refusal=None, tool_calls=tool_calls,
                                 model_extra={"reasoning_content": reasoning} if reasoning else {})


def _chunk(delta, finish=None, **meta):
    return types.SimpleNamespace(id="chatcmpl-t", model="fake", created=1, usage=None, **meta,
                                 choices=[types.SimpleNamespace(index=0, delta=delta, finish_reason=finish,
                                                                logprobs=None)])


def _msg(content: str, finish: str = "stop", reasoning: str = ""):
    """假的流式响应：思考、正文各拆成两段，最后一个片段带 finish_reason。"""
    chunks = []
    for text, key in ((reasoning, "reasoning"), (content, "content")):
        if text:
            half = len(text) // 2
            chunks += [_chunk(_delta(**{key: part})) for part in (text[:half], text[half:]) if part]
    return iter(chunks + [_chunk(_delta(), finish)])


class FakeClient:
    """按提示词种类返回预设回答；self_check_replies 依次用于每轮自检。"""

    def __init__(self, decision: dict, self_check_replies: list[dict], complexity="medium",
                 guard_replies: list[dict] | None = None, pick_replies: list[dict] | None = None):
        self.decision = decision
        self.pick_replies = list(pick_replies or [])
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
        if "只在这份名单里选一个" in last:
            return _msg(json.dumps(self.pick_replies.pop(0), ensure_ascii=False))
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
        mv, records = hooks.self_check(self.board, [], self.board.parse_san(first), self.legal, "medium")
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
        mv, records = hooks.hang_guard(self.board, [], self.board.parse_san("Bg5"), self.legal,
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
        mv, records = hooks.hang_guard(board, [], board.parse_san("Bxh7+"),
                                        list(player.legal_san_map(board)), [])
        self.assertEqual(board.san(mv), "Bxh7+")
        self.assertEqual(records[0]["outcome"], "kept_tactical")

    def test_positional_keep_without_line(self):
        board = chess.Board("rnbq1rk1/pppn1ppp/4p3/3pP3/1b1P4/2NB1N2/PPP2PPP/R1BQK2R w KQ - 0 7")
        llm._client = FakeClient({}, [], guard_replies=[
            {"verdict": "keep", "move": "Bxh7+", "kind": "positional", "line": "", "reason": "王翼被削弱"}])
        mv, records = hooks.hang_guard(board, [], board.parse_san("Bxh7+"),
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
        mv, records = hooks.hang_guard(self.board, [], self.board.parse_san("Re2"), self.legal, [])
        self.assertEqual(records, [])
        self.assertEqual(llm._client.prompts, [])


# 日志 20261010_012422：白第 24 步。候选 Qxe6、Bxa7 都丢子，守卫两轮 Rc2 → Qxd7 → Rc3，
# 旧逻辑把从没复查过的 Rc3 当"原着法"保持，Qa5xc3 白丢车
PLY47 = "4r1k1/p2n1pp1/2Q1bb1p/q1B1p3/4P3/3P1N1P/Pr2BPP1/2R1R1K1 w - - 1 24"
# 同一盘白第 26 步：Bd4 → Qxe6 → 想改回 Bd4 被拒，旧逻辑走了模型已放弃的 Qxe6；按规则只有 d4 不丢子
PLY51 = "4r1k1/p2n1pp1/2Q1bb1p/2B1p3/4P3/2qP1N1P/P3RPP1/6K1 w - - 0 26"


class HangGuardFallbackTest(unittest.TestCase):
    def run_guard(self, fen, first, replies, candidates, picks=()):
        board = chess.Board(fen)
        llm._client = FakeClient({}, [], guard_replies=replies, pick_replies=list(picks))
        mv, records = hooks.hang_guard(board, [], board.parse_san(first), list(player.legal_san_map(board)),
                                        candidates)
        return board.san(mv), records, llm._client

    def test_unchecked_last_change_is_not_kept(self):
        san, records, client = self.run_guard(PLY47, "Rc2", [change("Qxd7"), change("Rc3")],
                                              ["Qxe6", "Bxa7"], picks=[{"move": "a3", "reason": "安全"}])
        self.assertEqual(san, "a3")
        self.assertEqual(records[-1]["outcome"], "fallback_pick")
        self.assertIn("Rc3", client.prompts[-1])     # 名单前列出已查出丢子的 Rc3
        self.assertNotIn("Rc3,", client.prompts[-1].split("只有这些")[1])

    def test_invalid_pick_uses_least_loss_safe_move(self):
        board = chess.Board(PLY47)
        san, records, _ = self.run_guard(PLY47, "Rc2", [change("Qxd7"), change("Rc3")],
                                         ["Qxe6", "Bxa7"], picks=[{"move": "Rc3", "reason": "坚持"}])
        self.assertEqual(records[-1]["outcome"], "fallback_legal")
        self.assertLess(guard.material_risk(board, board.parse_san(san))["loss"], 2)

    def test_flip_back_uses_only_safe_legal_move(self):
        san, records, _ = self.run_guard(PLY51, "Bd4", [change("Qxe6"), change("Bd4")],
                                         ["Qxe6"], picks=[{"move": "d4", "reason": "唯一不丢子"}])
        self.assertEqual(san, "d4")

    def test_all_moves_lose_takes_least_loss(self):
        # 白马 h1 被困：走马被兵吃、走王被象吃，所有着法都净亏 3；Ng3 → Nf2 → 想改回 Ng3 被拒
        fen = "7k/8/8/3b4/7p/4p3/8/K6N w - - 0 1"
        board = chess.Board(fen)
        losses = {board.san(m): guard.material_risk(board, m)["loss"] for m in board.legal_moves}
        self.assertGreaterEqual(min(losses.values()), 2)
        san, records, _ = self.run_guard(fen, "Ng3", [change("Nf2"), change("Ng3")], [])
        self.assertEqual(losses[san], min(losses.values()))
        self.assertIn(san, ("Ng3", "Nf2"))  # 同分时优先模型提过的着法
        self.assertIn(records[-1]["outcome"], ("least_loss", "unresolved"))


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


class HookPipelineTest(unittest.TestCase):
    """run_pre_move 的调度：recheck 守卫在后面的 hook 改着后重跑，final hook 改着不触发重跑。"""

    def setUp(self):
        self.board = chess.Board()
        self.ctx = hooks.MoveContext(board=self.board, messages=[], legal_sans=[], candidates=[],
                                     complexity="", check_levels=[], levels=[], max_tokens=None)
        self.calls: list[str] = []

    def hook(self, name, to=None, **kw):
        def run(ctx, mv):
            self.calls.append(f"{name}:{self.board.san(mv)}")
            new = self.board.parse_san(to) if to else mv
            return new, [{"hook": name}] if new != mv else []
        return hooks.Hook(name, name, lambda ctx: True, run, **kw)

    def test_recheck_after_later_change(self):
        mv, recs = hooks.run_pre_move(self.ctx, self.board.parse_san("e4"),
                                      [self.hook("guard", recheck=True), self.hook("check", to="d4")])
        self.assertEqual(self.board.san(mv), "d4")
        self.assertEqual(self.calls, ["guard:e4", "check:e4", "guard:d4"])
        self.assertEqual(list(recs), ["check"])  # 没改着的 hook 没有记录

    def test_final_change_skips_recheck(self):
        mv, _ = hooks.run_pre_move(self.ctx, self.board.parse_san("e4"),
                                   [self.hook("guard", recheck=True), self.hook("mate", to="Nf3", final=True)])
        self.assertEqual(self.board.san(mv), "Nf3")
        self.assertEqual(self.calls, ["guard:e4", "mate:e4"])

    def test_disabled_hook_skipped(self):
        off = hooks.Hook("off", "off", lambda ctx: False, lambda ctx, mv: (mv, [{"x": 1}]))
        mv, recs = hooks.run_pre_move(self.ctx, self.board.parse_san("e4"), [off])
        self.assertEqual(recs, {})


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
                return _msg("", finish="length", reasoning=self.REASONING)
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


def _real_chunk(delta: dict, finish=None, usage=None):
    """用 openai 库真实的 ChatCompletionChunk，确认拼装逻辑对真实对象也成立。"""
    from openai.types.chat import ChatCompletionChunk
    return ChatCompletionChunk.model_validate({
        "id": "chatcmpl-real", "object": "chat.completion.chunk", "created": 7, "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if delta is not None else [],
        **({"usage": usage} if usage else {})})


def _status_error(code: int):
    import httpx
    from openai import InternalServerError
    return InternalServerError(f"Error code: {code}", body=None,
                               response=httpx.Response(code, request=httpx.Request("POST", "http://x")))


class StreamTest(unittest.TestCase):
    """流式片段拼装，以及断流 / 网关超时的处理。"""

    def setUp(self):
        self.sleep, llm.time.sleep = llm.time.sleep, lambda s: None
        self.salvage, llm.TRUNCATE_SALVAGE = llm.TRUNCATE_SALVAGE, "tail"

    def tearDown(self):
        llm.time.sleep = self.sleep
        llm.TRUNCATE_SALVAGE = self.salvage

    def use(self, create):
        llm._client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))

    def test_accumulates_everything(self):
        chunks = [
            _real_chunk({"role": "assistant", "reasoning_content": "先看 "}),
            _real_chunk({"reasoning_content": "e4。"}),
            _real_chunk({"content": "查一下"}),
            _real_chunk({"tool_calls": [{"index": 0, "id": "call_a", "type": "function",
                                         "function": {"name": "play_line", "arguments": '{"mo'}}]}),
            _real_chunk({"tool_calls": [{"index": 1, "id": "call_b", "type": "function",
                                         "function": {"name": "search_experience", "arguments": '{"q'}}]}),
            _real_chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ves": "e4 e5"}'}}]}),
            _real_chunk({"tool_calls": [{"index": 1, "function": {"arguments": 'uery": "x"}'}}]}),
            _real_chunk({}, finish="tool_calls"),
            _real_chunk(None, usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        ]
        self.use(lambda **kw: iter(chunks))
        resp = llm.complete(model="m", messages=[])
        choice = resp.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual(resp.id, "chatcmpl-real")
        self.assertEqual(resp.usage.total_tokens, 15)
        msg = choice.message
        self.assertEqual(msg.content, "查一下")
        self.assertEqual(llm.reasoning_of(msg), "先看 e4。")
        self.assertEqual([tc.id for tc in msg.tool_calls], ["call_a", "call_b"])
        self.assertEqual(json.loads(msg.tool_calls[0].function.arguments), {"moves": "e4 e5"})
        self.assertEqual(json.loads(msg.tool_calls[1].function.arguments), {"query": "x"})
        self.assertEqual(msg.tool_calls[1].model_dump()["function"]["name"], "search_experience")

    def test_reasoning_field_name_variant(self):
        self.use(lambda **kw: iter([_real_chunk({"reasoning": "想"}), _real_chunk({"reasoning": "完"}),
                                    _real_chunk({"content": "{}"}, finish="stop")]))
        msg = llm.complete(model="m", messages=[]).choices[0].message
        self.assertEqual(llm.reasoning_of(msg), "想完")

    def test_request_is_streamed(self):
        calls = []
        self.use(lambda **kw: calls.append(kw) or _msg("{}"))
        llm.llm_call([{"role": "user", "content": "x"}], levels=["low"])
        self.assertTrue(calls[0]["stream"])

    def test_gateway_timeout_retries_then_steps_down(self):
        calls = []

        def create(**kw):
            calls.append(kw)
            if len(calls) <= 2:  # 第一档：首次请求 + 1 次重试都是 524
                raise _status_error(524)
            return _msg(json.dumps({"move": "e4"}))

        self.use(create)
        msg, level = llm.llm_call([{"role": "user", "content": "x"}], levels=["high", "low"])
        self.assertEqual(level, "low")
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[2]["reasoning_effort"], "low")

    def test_client_error_is_not_retried(self):
        calls = []

        def create(**kw):
            calls.append(kw)
            import httpx
            from openai import BadRequestError
            raise BadRequestError("bad", body=None,
                                  response=httpx.Response(400, request=httpx.Request("POST", "http://x")))

        self.use(create)
        with self.assertRaises(Exception):
            llm.llm_call([{"role": "user", "content": "x"}], levels=["high", "low"])
        self.assertEqual(len(calls), 1)

    def test_interrupted_stream_salvages_partial_reasoning(self):
        import httpx
        calls = []

        def broken():
            yield _chunk(_delta(reasoning="分析到一半：倾向 Nf3，"))
            yield _chunk(_delta(reasoning="d4 也可以"))
            raise httpx.ReadTimeout("stalled")

        def create(**kw):
            calls.append(kw)
            return broken() if len(calls) == 1 else _msg(json.dumps({"move": "Nf3"}))

        self.use(create)
        msg, level = llm.llm_call([{"role": "user", "content": "x"}], levels=["high", "low"])
        self.assertEqual(level, "low")
        retry = calls[1]["messages"][-1]["content"]
        self.assertIn("连接中断", retry)
        self.assertIn("倾向 Nf3，d4 也可以", retry)  # 断开前收到的思考完整带回
        self.assertIn("中途断开", msg.reasoning_content)

    def test_missing_finish_reason(self):
        self.use(lambda **kw: iter([_chunk(_delta(content='{"move": "e4"}'))]))
        self.assertEqual(llm.complete(model="m", messages=[]).choices[0].finish_reason, "stop")
        self.use(lambda **kw: iter([_chunk(_delta(reasoning="想了一半"))]))
        with self.assertRaises(llm.StreamInterrupted) as ctx:
            llm.complete(model="m", messages=[])
        self.assertEqual(llm.reasoning_of(ctx.exception.partial.choices[0].message), "想了一半")

    def test_safe_llm_move_falls_back_when_all_levels_fail(self):
        def create(**kw):
            raise _status_error(524)

        self.use(create)
        board = chess.Board()
        uci, *_ = player.safe_llm_move(board, 1, None, None)
        self.assertIn(chess.Move.from_uci(uci), board.legal_moves)


class LineMemoryTest(unittest.TestCase):
    """上一步主变 pv 跨步保存：连杀 / 唯一应着直接走，按主变应着时提示续走，偏离时提示预期 vs 实际。"""

    def setUp(self):
        player._plans.clear()

    def first_move(self, board: chess.Board, decision: dict) -> chess.Board:
        """我方按 decision 走一步，返回走之前的局面（供下一步的 prev_board）。"""
        llm._client = FakeClient(decision, [keep(decision["move"])])
        uci, *_ = player.get_llm_move(board, board.ply() + 1, None, None)
        board.push_uci(uci)
        return board.copy()

    def test_only_reply_plays_next_without_llm(self):
        # Qd8+ 之后黑方只有 Kf7 一个合法应着：直接走主变里的 Qd5+（捉双抽车），不再调 LLM
        board = chess.Board("6k1/6pp/8/8/8/8/r5PP/3Q2K1 w - - 0 1")
        prev = self.first_move(board, {"think": "x", "pv": ["Qd8+", "Kf7", "Qd5+", "Ke7", "Qxa2"],
                                       "pv_goal": "将军后 Qd5+ 捉双，抽车", "move": "Qd8+"})
        board.push_san("Kf7")
        llm._client = FakeClient({"move": "Kf1"}, [])
        uci, think, _, obs = player.get_llm_move(board, board.ply() + 1, prev, "g8f7")
        self.assertEqual(board.san(chess.Move.from_uci(uci)), "Qd5+")
        self.assertEqual(llm._client.prompts, [])
        self.assertIn("唯一", think)
        self.assertEqual(obs["pv"], ["Qd5+", "Ke7", "Qxa2"])
        self.assertEqual(obs["pv_goal"], "将军后 Qd5+ 捉双，抽车")

    def test_mate_line_executed_by_rule(self):
        # 主变摆到底是将杀：对方按主变应着（虽然不是唯一应着）就直接走
        board = chess.Board()
        for san in ["e4", "e5", "Bc4", "Nc6"]:
            board.push_san(san)
        prev = self.first_move(board, {"think": "x", "pv": ["Qh5", "Nf6", "Qxf7+"],
                                       "pv_goal": "f7 只有王保护，Qxf7 杀", "move": "Qh5"})
        self.assertTrue(player._plans[chess.WHITE]["mate"])
        board.push_san("Nf6")
        llm._client = FakeClient({"move": "a3"}, [])
        uci, *_ = player.get_llm_move(board, board.ply() + 1, prev, "g8f6")
        self.assertEqual(uci, "h5f7")
        self.assertEqual(llm._client.prompts, [])

    def test_mate_line_deviation_rethinks(self):
        board = chess.Board()
        for san in ["e4", "e5", "Bc4", "Nc6"]:
            board.push_san(san)
        prev = self.first_move(board, {"think": "x", "pv": ["Qh5", "Nf6", "Qxf7+"],
                                       "pv_goal": "f7 只有王保护，Qxf7 杀", "move": "Qh5"})
        board.push_san("g6")
        llm._client = FakeClient({"think": "x", "pv": ["Qf3"], "move": "Qf3"}, [keep("Qf3")])
        uci, *_ = player.get_llm_move(board, board.ply() + 1, prev, "g7g6")
        self.assertEqual(uci, "h5f3")
        decision_prompt = next(p for p in llm._client.prompts if "【当前局面，轮到你走】" in p)
        self.assertIn("你预期对方走 Nf6，对方实际走了 g6", decision_prompt)
        self.assertIn("f7 只有王保护", decision_prompt)
        self.assertTrue(any("只判断当前局面的复杂度" in p for p in llm._client.prompts))

    def test_followed_line_asks_model_to_confirm(self):
        # 对方按主变应着、但既不是连杀也不是唯一应着：提示模型续走，跳过复杂度判断
        board = chess.Board()
        prev = self.first_move(board, {"think": "x", "pv": ["e4", "e5", "Nf3"],
                                       "pv_goal": "出子攻击 e5", "move": "e4"})
        board.push_san("e5")
        llm._client = FakeClient({"think": "x", "pv": ["Nf3", "Nc6"], "move": "Nf3"}, [keep("Nf3")])
        uci, _, _, obs = player.get_llm_move(board, 3, prev, "e7e5")
        self.assertEqual(uci, "g1f3")
        self.assertFalse(any("只判断当前局面的复杂度" in p for p in llm._client.prompts))
        decision_prompt = next(p for p in llm._client.prompts if "【当前局面，轮到你走】" in p)
        self.assertIn("计划中的下一步是 Nf3", decision_prompt)
        self.assertIn("出子攻击 e5", decision_prompt)
        self.assertIn("对方按上一步主变应着", obs["complexity_reason"])

    def hanging_plan_board(self) -> chess.Board:
        """日志 20261010_012422 白第 25 步主变 Rxe2 Qxc3 Qxd7：对方按主变应了 Qxc3，
        但 e6 象保护 d7，Qxd7 会被 Bxd7 吃后。"""
        board = chess.Board(PLY47)
        for san in ["Rc3", "Rxe2", "Rxe2"]:
            board.push_san(san)
        moves = [m.uci() for m in board.move_stack]
        line = [board.peek().uci(), "a5c3", "c6d7"]
        player._plans[chess.WHITE] = {"moves": moves, "strategy": "兑车后吃马", "line": line, "mate": False,
                                      "goal": "兑车后 Qxd7 得子"}
        board.push_san("Qxc3")
        return board

    def test_hanging_next_move_is_rethought(self):
        board = self.hanging_plan_board()
        llm._client = FakeClient({"think": "x", "move": "d4"}, [keep("d4")])
        uci, _, _, obs = player.get_llm_move(board, board.ply() + 1, None, "a5c3")
        self.assertEqual(uci, "d3d4")
        # 不降档：照常做复杂度判断
        self.assertTrue(any("只判断当前局面的复杂度" in p for p in llm._client.prompts))
        self.assertNotIn("对方按上一步主变应着", obs.get("complexity_reason", ""))
        decision_prompt = next(p for p in llm._client.prompts if "【当前局面，轮到你走】" in p)
        self.assertIn("计划中的下一步是 Qxd7", decision_prompt)
        self.assertIn("Bxd7", decision_prompt)
        self.assertNotIn("不必从头重新计算", decision_prompt)

    def test_hanging_next_move_not_played_as_only_reply(self):
        board = self.hanging_plan_board()
        original = player.line_progress
        player.line_progress = lambda b, plan: dict(original(b, plan), only_reply=True)
        try:
            llm._client = FakeClient({"think": "x", "move": "d4"}, [keep("d4")])
            uci, *_ = player.get_llm_move(board, board.ply() + 1, None, "a5c3")
        finally:
            player.line_progress = original
        self.assertEqual(uci, "d3d4")
        self.assertTrue(llm._client.prompts)

    def test_invalid_pv_is_truncated(self):
        board = chess.Board()
        self.first_move(board, {"think": "x", "pv": ["e4", "e5", "Ke3", "Nf3"], "move": "e4"})
        self.assertEqual(player._plans[chess.WHITE]["line"], ["e2e4", "e7e5"])


LOST_GAME = ("d4 d5 Bf4 Nc6 c3 e6 Nf3 Nf6 Qc2 Be7 Bg3 O-O e3 Bd6 Nbd2 Bxg3 a3 Bd6 h4 e5 g4 Nxg4 Ng5"
             .split())  # 实战：黑方接着走 h6??，白 Qh7#


def lost_position() -> chess.Board:
    board = chess.Board()
    for san in LOST_GAME:
        board.push_san(san)
    return board


class MateGuardTest(unittest.TestCase):
    def setUp(self):
        player._plans.clear()

    def client(self, decision: dict, guard_moves: list[str]) -> FakeClient:
        """决策回答 decision；将杀守卫的每轮依次改选 guard_moves。"""
        client = FakeClient(decision, [keep(decision["move"])] * 3)
        original = client._create
        replies = list(guard_moves)

        def create(**kw):
            last = kw["messages"][-1]["content"]
            if "一步就能将杀" in last:
                client.prompts.append(last)
                return _msg(json.dumps({"move": replies.pop(0), "reason": "防杀"}, ensure_ascii=False))
            return original(**kw)

        client.chat.completions.create = create
        llm._client = client
        return client

    def test_rule_checks(self):
        board = lost_position()
        self.assertTrue(player.mate_threat(board))
        self.assertFalse(player.mate_in_one(board))
        self.assertTrue(player.allows_mate(board, board.parse_san("h6")))
        self.assertFalse(player.allows_mate(board, board.parse_san("g6")))

    def test_lost_game_h6_is_replaced(self):
        board = lost_position()
        client = self.client({"think": "x", "candidates": [{"move": "h6"}], "pv": ["h6"], "move": "h6"},
                             ["g6"])
        uci, _, _, obs = player.get_llm_move(board, 24, None, None)
        self.assertEqual(board.san(chess.Move.from_uci(uci)), "g6")
        self.assertEqual(obs["mate_guard"][0]["changed_to"], "g6")
        guard_prompt = next(p for p in client.prompts if "一步就能将杀" in p)
        decision_prompt = next(p for p in client.prompts if "【当前局面，轮到你走】" in p)
        # 只告诉存在一步杀，不泄露对方的杀着
        for prompt in (guard_prompt, decision_prompt):
            self.assertNotIn("Qxh7", prompt)
            self.assertNotIn("Qh7", prompt)
        self.assertIn("对方有一步将杀", decision_prompt)

    def test_insisting_falls_back_to_safe_candidate(self):
        board = lost_position()
        self.client({"think": "x", "candidates": [{"move": "h6"}, {"move": "Nf6"}], "pv": ["h6"],
                     "move": "h6"}, ["h6", "h6"])
        uci, _, _, obs = player.get_llm_move(board, 24, None, None)
        self.assertEqual(board.san(chess.Move.from_uci(uci)), "Nf6")
        self.assertEqual(obs["mate_guard"][-1]["outcome"], "fallback")

    def test_safe_move_makes_no_extra_call(self):
        board = lost_position()
        client = self.client({"think": "x", "pv": ["g6"], "move": "g6"}, [])
        uci, *_ = player.get_llm_move(board, 24, None, None)
        self.assertEqual(uci, "g7g6")
        self.assertFalse(any("一步就能将杀" in p for p in client.prompts))

    def test_own_mate_in_one_skips_book_and_fast_mode(self):
        board = chess.Board()
        for san in ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"]:
            board.push_san(san)
        self.client({"think": "x", "pv": ["Qxf7+"], "move": "Qxf7+"}, [])
        original_lookup = player.book.lookup
        player.book.lookup = lambda b: {"uci": "a2a3", "san": "a3", "n": 9, "count": 9, "avg_cp": 50,
                                        "options": 1}
        player.OPENING_FAST_MOVES = 8
        try:
            uci, _, _, obs = player.get_llm_move(board, 7, None, None)
        finally:
            player.book.lookup = original_lookup
            player.OPENING_FAST_MOVES = 0
        self.assertEqual(uci, "h5f7")
        self.assertNotEqual(obs.get("complexity_reason"), "开局快速模式")
        prompt = next(p for p in llm._client.prompts if "【当前局面，轮到你走】" in p)
        self.assertNotIn("Qxf7#", prompt)  # 我方的一步杀不告诉模型

    def test_pv_shortcut_not_taken_into_mate(self):
        # 主变下一步会被一步杀：不直接走，交给模型（再由守卫拦下）
        board = lost_position()
        board.pop()  # 回到白方 Ng5 之前，让黑方的"主变"从 ...Nxg4 之前开始
        board.pop()
        self.client({"think": "x", "pv": ["Nxg4", "Ng5", "h6"], "move": "Nxg4"}, [])
        uci, *_ = player.get_llm_move(board, 22, None, None)
        self.assertEqual(uci, "f6g4")
        board.push_uci(uci)
        board.push_san("Ng5")
        # Ng5 不是白方唯一应着，本来就不会直接走；把它伪装成唯一应着，确认安全检查仍会拦住
        original = player.line_progress

        def forced(b, plan):
            prog = original(b, plan)
            return dict(prog, only_reply=True) if prog else prog

        player.line_progress = forced
        try:
            client = self.client({"think": "x", "pv": ["g6"], "move": "g6"}, [])
            uci, *_ = player.get_llm_move(board, 24, None, None)
        finally:
            player.line_progress = original
        self.assertEqual(uci, "g7g6")
        self.assertTrue(client.prompts)


if __name__ == "__main__":
    unittest.main()
