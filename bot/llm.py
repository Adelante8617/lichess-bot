"""LLM 调用：OpenAI 兼容客户端（流式拼装）、思考档位阶梯、截断 / 断流后的降档补救、JSON 提取。"""
import json
import re
import time
from types import SimpleNamespace

import httpx
from openai import APIConnectionError, APIError, APIStatusError, OpenAI
from openai.types.chat import ChatCompletionMessage

from .config import (EFFORT_LEVELS, LLM_API_KEY, LLM_BASE_URL, LLM_EXTRA_BODY, LLM_MAX_TOKENS,
                     LLM_RETRIES, LLM_STREAM, LLM_TEMPERATURE, LLM_TIMEOUT, MODEL, THINK_LADDER,
                     TRUNCATE_REASONING_TAIL, TRUNCATE_SALVAGE, TRUNCATE_SUMMARY_MAX_TOKENS)
from .live import live

_client = None


def client() -> OpenAI:
    """首次用到时才创建客户端，导入本模块不需要 API key。
    SDK 自带的重试关掉（它会把 524 这类超时原样重发、白等几分钟），由 complete() / llm_call() 控制。"""
    global _client
    if _client is None:
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, max_retries=0,
                         timeout=httpx.Timeout(LLM_TIMEOUT, connect=15.0))
    return _client


class StreamInterrupted(Exception):
    """流式响应收到部分数据后断开。partial：已收到部分拼成的响应（结构同 complete() 的返回）。"""

    def __init__(self, partial, cause):
        super().__init__(f"流式响应中途断开：{cause}")
        self.partial = partial
        self.cause = cause


def is_transient(e: Exception) -> bool:
    """值得重试 / 降档再试的错误：连接错误、超时、429、5xx（含网关 524）。"""
    if isinstance(e, (APIConnectionError, httpx.TransportError)):
        return True
    return isinstance(e, APIStatusError) and (e.status_code == 429 or e.status_code >= 500)


class _StreamAccumulator:
    """把流式片段拼回成与非流式相同结构的响应：
    id / model / created / system_fingerprint / usage，以及每个 choice 的 role、content、refusal、
    tool_calls（按 index 合并 id / name / arguments 片段）、finish_reason、logprobs，
    和 delta 里供应商私有的字段（reasoning_content / reasoning 等：字符串逐段拼接，其他取最后一个值）。"""

    META = ("id", "model", "created", "system_fingerprint", "service_tier")

    def __init__(self):
        self.meta: dict = {}
        self.usage = None
        self.choices: dict[int, dict] = {}
        self.received = False

    def add(self, chunk):
        self.received = True
        for k in self.META:
            if getattr(chunk, k, None) is not None:
                self.meta[k] = getattr(chunk, k)
        if getattr(chunk, "usage", None):
            self.usage = chunk.usage
        for ch in getattr(chunk, "choices", None) or []:
            c = self.choices.setdefault(getattr(ch, "index", 0) or 0, {
                "role": "assistant", "content": [], "refusal": [], "tool_calls": {},
                "finish_reason": None, "logprobs": [], "extra": {}})
            if getattr(ch, "finish_reason", None):
                c["finish_reason"] = ch.finish_reason
            logprobs = getattr(ch, "logprobs", None)
            if logprobs is not None and getattr(logprobs, "content", None):
                c["logprobs"].extend(logprobs.content)
            delta = getattr(ch, "delta", None)
            if delta is None:
                continue
            if getattr(delta, "content", None):
                c["content"].append(delta.content)
            if getattr(delta, "refusal", None):
                c["refusal"].append(delta.refusal)
            for tc in getattr(delta, "tool_calls", None) or []:
                self._add_tool_call(c["tool_calls"], tc)
            extra = dict(getattr(delta, "model_extra", None) or {})
            for k in ("reasoning_content", "reasoning"):  # SDK 若把它们做成正式字段，model_extra 里就没有
                if k not in extra and isinstance(getattr(delta, k, None), str):
                    extra[k] = getattr(delta, k)
            for k, v in extra.items():
                if isinstance(v, str):
                    c["extra"][k] = c["extra"].get(k, "") + v
                elif v is not None:
                    c["extra"][k] = v

    @staticmethod
    def _add_tool_call(calls: dict, tc):
        idx = getattr(tc, "index", None)
        tid = getattr(tc, "id", None)
        if idx is None:  # 个别供应商不给 index：有 id 按 id 归并，没有就接到最后一个上
            idx = next((i for i, x in calls.items() if tid and x["id"] == tid), None)
            if idx is None:
                idx = (max(calls) if calls else 0) if not tid else (max(calls) + 1 if calls else 0)
        e = calls.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        if tid:
            e["id"] = tid
        if getattr(tc, "type", None):
            e["type"] = tc.type
        fn = getattr(tc, "function", None)
        if fn is not None:
            name = getattr(fn, "name", None)
            if name and name != e["function"]["name"]:  # 名字一般只来一次；分段来的就拼起来
                e["function"]["name"] += name
            if getattr(fn, "arguments", None):
                e["function"]["arguments"] += fn.arguments

    def result(self):
        choices = []
        for idx in sorted(self.choices) or [0]:
            c = self.choices.get(idx) or {"role": "assistant", "content": [], "refusal": [], "tool_calls": {},
                                          "finish_reason": None, "logprobs": [], "extra": {}}
            msg = {**c["extra"], "role": "assistant", "content": "".join(c["content"]) or None}
            if c["refusal"]:
                msg["refusal"] = "".join(c["refusal"])
            if c["tool_calls"]:
                msg["tool_calls"] = [dict(tc, id=tc["id"] or f"call_{i}")
                                     for i, tc in sorted(c["tool_calls"].items())]
            if not msg.get("reasoning_content") and isinstance(msg.get("reasoning"), str):
                msg["reasoning_content"] = msg["reasoning"]  # 统一放到 reasoning_content，reasoning_of 只看这里
            choices.append(SimpleNamespace(index=idx, message=ChatCompletionMessage.model_validate(msg),
                                           finish_reason=c["finish_reason"], logprobs=c["logprobs"] or None))
        return SimpleNamespace(object="chat.completion", **{k: self.meta.get(k) for k in self.META},
                               choices=choices, usage=self.usage)


def complete(**kw):
    """统一的 chat completion 调用，返回结构与非流式响应相同（resp.choices[0].message / finish_reason）。
    LLM_STREAM=1 时用流式请求，把全部片段拼回完整响应。
    还没收到任何数据就失败的临时性错误重试 LLM_RETRIES 次；收到部分数据后断开则抛 StreamInterrupted，
    带上已收到的部分（思考过程等），由调用方决定怎么补救。"""
    for attempt in range(LLM_RETRIES + 1):
        acc = _StreamAccumulator()
        try:
            if not LLM_STREAM:
                return client().chat.completions.create(**kw)
            for chunk in client().chat.completions.create(stream=True, **kw):
                acc.add(chunk)
        except (APIError, httpx.HTTPError) as e:
            if acc.received:
                raise StreamInterrupted(acc.result(), e) from e
            if not is_transient(e) or attempt == LLM_RETRIES:
                raise
            print(f"[LLM] 请求失败，2 秒后重试（{attempt + 1}/{LLM_RETRIES}）：{e}")
            time.sleep(2)
            continue
        resp = acc.result()
        first = resp.choices[0]
        if first.finish_reason is None:
            # 流正常结束却没有结束原因：有正文 / 工具调用就当作完整回答，否则视为断流
            if first.message.content or first.message.tool_calls:
                print("[LLM] 流式响应没有 finish_reason，按 stop 处理")
                first.finish_reason = "stop"
            else:
                raise StreamInterrupted(resp, "流结束时没有 finish_reason，也没有正文")
        return resp


def extract_json(text: str) -> dict | None:
    """从模型输出里取出第一个能解析的 JSON 对象（容忍前后缀文字、多个花括号）。"""
    decoder = json.JSONDecoder()
    pos = text.find("{")
    while pos != -1:
        try:
            obj, _ = decoder.raw_decode(text, pos)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        pos = text.find("{", pos + 1)
    return None


def effort_kwargs(level: str) -> dict:
    """思考档位 → 请求参数（reasoning_effort + extra_body.thinking），与 LLM_EXTRA_BODY 合并。"""
    kw: dict = {}
    extra = dict(LLM_EXTRA_BODY)
    if level in EFFORT_LEVELS:
        kw["reasoning_effort"] = level
        extra["thinking"] = {"type": "enabled"}
    elif level == "off":
        extra["thinking"] = {"type": "disabled"}
    if extra:
        kw["extra_body"] = extra
    return kw


def think_ladder(start: str | None = None) -> list[str]:
    """本步使用的档位阶梯：从 start 开始沿 THINK_LADDER 往下降；start 不在阶梯中则只用它一档。"""
    if not start:
        return THINK_LADDER
    if start in THINK_LADDER:
        return THINK_LADDER[THINK_LADDER.index(start):]
    return [start]


def cap_ladder(levels: list[str], cap: str) -> list[str]:
    """把档位阶梯封顶到 cap：levels 起始档高于 cap（按 THINK_LADDER 顺序）时改从 cap 开始。"""
    if not cap or (cap in THINK_LADDER and levels and levels[0] in THINK_LADDER
                   and THINK_LADDER.index(levels[0]) >= THINK_LADDER.index(cap)):
        return levels
    return think_ladder(cap)


def llm_call(messages: list, tools: list | None = None, levels: list[str] | None = None,
             max_tokens: int | None = None):
    """下棋阶段统一的 LLM 调用，返回 (message, 实际使用的档位)。从 levels[0] 开始：
    - 思考把 max_tokens 用光、正文为空，或流式响应中途断开：带回已有思考的要点（或末尾），换下一档直接要结论；
    - 请求失败（重试后仍是连接错误 / 超时 / 5xx）且没收到任何数据：换下一档重新请求。
    已经是最后一档时，截断照常返回，出错则抛出，由上层保底。"""
    levels = levels or THINK_LADDER[:1]
    max_tokens = max_tokens or LLM_MAX_TOKENS
    msgs, use_tools = messages, tools
    truncated: list[str] = []  # 被截断的思考，并入最终消息的 reasoning，日志和观战页才看得到
    for i, level in enumerate(levels):
        last = i == len(levels) - 1
        kw = dict(model=MODEL, messages=msgs, temperature=LLM_TEMPERATURE,
                  max_tokens=max_tokens, **effort_kwargs(level))
        if use_tools:
            kw["tools"] = use_tools
        print(f"[LLM] effort={level}")
        try:
            choice = complete(**kw).choices[0]
        except StreamInterrupted as e:
            if last:
                raise
            partial = e.partial.choices[0].message
            reasoning = reasoning_of(partial)
            if partial.content:  # 断在正文中途：已写出的部分正文也算分析的一部分
                reasoning += f"\n[已输出的部分正文]\n{partial.content}"
            print(f"[WARN] 流式响应中途断开（effort={level}，已收到思考 {len(reasoning)} 字）：{e.cause}")
            msgs = _salvage(messages, reasoning, level, levels[i + 1], truncated,
                            "你刚才的回答因连接中断没有完成，没有给出最终答案。\n", "中途断开")
            use_tools = None
            continue
        except Exception as e:
            if last or not is_transient(e):
                raise
            print(f"[WARN] 请求失败（effort={level}）：{e}，降档到 {levels[i + 1]} 重新请求")
            live.stage(f"请求失败，降档到 {levels[i + 1]}")
            continue
        msg = choice.message
        if choice.finish_reason != "length":
            return _merge_truncated(msg, truncated), level
        print(f"[WARN] 模型输出达到 max_tokens={max_tokens} 被截断（effort={level}）")
        if (msg.content or "").strip() or msg.tool_calls or last:
            return _merge_truncated(msg, truncated), level
        # 正文为空：把已有分析（压缩要点或思考末尾）带回，降一档、不给工具，直接要最终 JSON
        msgs = _salvage(messages, reasoning_of(msg), level, levels[i + 1], truncated,
                        "你刚才的思考过长被截断，没有给出最终答案。\n", "被截断")
        use_tools = None
    return msg, level


def _salvage(messages: list, reasoning: str, level: str, next_level: str, truncated: list[str],
             note: str, why: str) -> list:
    """思考没能给出答案（被截断 / 断流）时，把已有分析整理后带给下一档，返回下一档的 messages。
    truncated 里追加给日志 / 观战页看的记录。"""
    truncated.append(f"[effort={level} 的思考，{why}]\n{reasoning}")
    ask = "不要再展开新的计算，直接根据已有分析选定着法，按系统提示输出完整 JSON。"
    summary, dropped = summarize_reasoning(reasoning) if TRUNCATE_SALVAGE == "summary" else ("", [])
    if dropped:  # 只记进日志 / 观战页，不给下棋模型
        truncated.append("[要点中被删除的行（含原文未出现的着法）]\n" + "\n".join(dropped))
    if summary:
        truncated.append(f"[被截断思考的要点整理]\n{summary}")
        note += f"以下是你此前思考的要点整理，由程序自动压缩，可能有错：\n{summary}\n\n"
        ask = ("不要盲目采信要点：选定着法前，对照上面的棋盘核实你要走的这一步"
               "（落点是否被攻击、有无保护、是否送子），以及要点里支撑这步的结论；"
               "要点与棋盘不符时以棋盘为准。只核实这一步，不要重新展开全面分析，"
               "然后按系统提示输出完整 JSON。")
    elif TRUNCATE_SALVAGE != "none" and TRUNCATE_REASONING_TAIL > 0 and reasoning:
        note += f"以下是你思考的最后部分：\n{reasoning[-TRUNCATE_REASONING_TAIL:]}\n\n"
    print(f"[SALVAGE] 思考{why}（{len(reasoning)} 字），降档 {level} -> {next_level} 直接要结论")
    live.stage(f"思考{why}，降档到 {next_level}")
    return messages + [{"role": "user", "content": note + ask}]


SUMMARY_PROMPT = """下面是一段对国际象棋局面的思考过程，因为太长在中途被截断了。
请把它压缩成要点，供思考者据此直接选定着法。只整理原文已有的内容，不要补充新的分析或计算。

按下面几项列出（原文没涉及的项写"无"）：
1. 考虑过的候选着法：每个一行，写清原文对它的结论（可行 / 被否决及原因 / 未算完）
2. 发现的威胁与战术：对方的威胁、我方的战术机会、悬空或被攻击的子
3. 目前倾向的着法及理由；若原文还没倾向，写最后正在分析的着法和进展

必须忠实于原文：
- 每个候选的结论必须是原文自己得出的结论，原文没算完就写"未算完"，不要替原文下结论或改动结论
- 原文若已明确表态要走某步（如 "I'll play X"、"final: X"、"决定走 X"），第 3 项必须照写这步，
  不能换成别的着法，也不能写成"无倾向"；表态后又犹豫的，写最后一次表态的着法，并注明在犹豫什么
- 只能写原文出现过的着法，并沿用原文的写法（SAN，如 Nxe6、d8=Q），不要自己推出新的变化或着法
  （原文没出现过的着法会被程序整行删除）

总共不超过 400 字。

【被截断的思考】
{reasoning}"""

# 要点里可核对的着法：带棋子字母 / 吃子 / 升变 / 易位。单纯的兵步（如 g5）与格子名分不开，不检查
MOVE_RE = re.compile(r"(?<![A-Za-z0-9])(O-O(?:-O)?|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]"
                     r"|[a-h]x[a-h][1-8](?:=?[QRBN])?|[a-h][18]=?[QRBN])(?![a-z0-9])")


def _move_key(san: str) -> str:
    """忽略吃子符号与消歧字母比较：Nfxd4 与 Nxd4 视为同一步。"""
    if san.startswith("O-O") or san[0] not in "KQRBN":
        return san.replace("x", "").replace("=", "")
    return san[0] + re.findall(r"[a-h][1-8]", san)[-1]


def drop_unseen_moves(summary: str, reasoning: str) -> tuple[str, list[str]]:
    """删掉要点里含原文没出现过的着法的行，返回 (保留的要点, 被删的行)。"""
    seen = {_move_key(m) for m in MOVE_RE.findall(reasoning)}
    kept, dropped = [], []
    for line in summary.splitlines():
        unseen = [m for m in MOVE_RE.findall(line) if _move_key(m) not in seen]
        (dropped if unseen else kept).append(f"{line}（原文未出现：{'、'.join(unseen)}）" if unseen else line)
    return "\n".join(kept).strip(), dropped


def summarize_reasoning(reasoning: str) -> tuple[str, list[str]]:
    """把被截断的思考压缩成要点（不思考的独立调用），并删掉含原文没出现过的着法的行。
    返回 (要点, 被删的行)；失败或为空时要点为空串，由调用方退回带末尾。"""
    if not reasoning.strip():
        return "", []
    live.stage("整理被截断思考的要点")
    try:
        resp = complete(
            model=MODEL, temperature=LLM_TEMPERATURE, max_tokens=TRUNCATE_SUMMARY_MAX_TOKENS,
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(reasoning=reasoning)}],
            **effort_kwargs("off"))
        summary = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # 压缩只是补救手段，出错不能让这一步棋失败
        print(f"[SALVAGE] 要点整理失败：{e}")
        return "", []
    summary, dropped = drop_unseen_moves(summary, reasoning)
    print(f"[SALVAGE] 要点整理（{len(reasoning)} 字 -> {len(summary)} 字）：{summary[:200]}")
    for line in dropped:
        print(f"[SALVAGE] 删除含原文未出现着法的行：{line}")
    return summary, dropped


def _merge_truncated(msg, truncated: list[str]):
    """把之前被截断的思考拼到最终消息的 reasoning 前面。"""
    if truncated:
        merged = "\n\n".join(truncated + [f"[降档后的思考]\n{reasoning_of(msg)}"])
        try:
            msg.reasoning_content = merged
        except (AttributeError, ValueError):
            pass
    return msg


def reasoning_of(msg) -> str:
    """思考模型返回的推理过程（deepseek 等放在 reasoning_content），没有则为空串。"""
    return (getattr(msg, "reasoning_content", None)
            or (getattr(msg, "model_extra", None) or {}).get("reasoning_content") or "")
