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
                   "COMPLEXITY_CHECK": "1", "THINK_LADDER": "default", "MATERIAL_LEAD_SKIP": "8",
                   "BOARD_RELATIONS": "0", "ANALYSIS_BOARD": "0", "PLAN_MEMORY": "1"})

import chess  # noqa: E402

from bot import llm, player  # noqa: E402


def _msg(content: str, finish: str = "stop"):
    message = types.SimpleNamespace(content=content, tool_calls=None, reasoning_content="",
                                    model_extra={})
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason=finish)])


class FakeClient:
    """按提示词种类返回预设回答；self_check_replies 依次用于每轮自检。"""

    def __init__(self, decision: dict, self_check_replies: list[dict], complexity="medium"):
        self.decision = decision
        self.self_check_replies = list(self_check_replies)
        self.complexity = complexity
        self.prompts: list[str] = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        last = kw["messages"][-1]["content"]
        self.prompts.append(last)
        if "只判断当前局面的复杂度" in last:
            return _msg(json.dumps({"complexity": self.complexity, "reason": "测试"}, ensure_ascii=False))
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


class GetMoveTest(unittest.TestCase):
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

    def test_illegal_then_fallback(self):
        board = chess.Board()
        llm._client = FakeClient({"move": "Ke2"}, [])
        uci, *_ = player.get_llm_move(board, 1, None, None)
        self.assertIn(chess.Move.from_uci(uci), board.legal_moves)


if __name__ == "__main__":
    unittest.main()
