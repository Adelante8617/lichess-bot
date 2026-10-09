"""LLM 调用：OpenAI 兼容客户端、思考档位阶梯、截断后的降档补救、JSON 提取。"""
import json

from openai import OpenAI

from .config import (EFFORT_LEVELS, LLM_API_KEY, LLM_BASE_URL, LLM_EXTRA_BODY, LLM_MAX_TOKENS,
                     LLM_TEMPERATURE, MODEL, THINK_LADDER, TRUNCATE_REASONING_TAIL,
                     TRUNCATE_SALVAGE, TRUNCATE_SUMMARY_MAX_TOKENS)
from .live import live

_client = None


def client() -> OpenAI:
    """首次用到时才创建客户端，导入本模块不需要 API key。"""
    global _client
    if _client is None:
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client


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
    """下棋阶段统一的 LLM 调用，返回 (message, 实际使用的档位)。
    从 levels[0] 开始；若思考把 max_tokens 用光、正文为空，则带回思考要点（或末尾）、换下一档直接要结论。"""
    levels = levels or THINK_LADDER[:1]
    max_tokens = max_tokens or LLM_MAX_TOKENS
    msgs, use_tools = messages, tools
    truncated: list[str] = []  # 被截断的思考，并入最终消息的 reasoning，日志和观战页才看得到
    for i, level in enumerate(levels):
        kw = dict(model=MODEL, messages=msgs, temperature=LLM_TEMPERATURE,
                  max_tokens=max_tokens, **effort_kwargs(level))
        if use_tools:
            kw["tools"] = use_tools
        print(f"[LLM] effort={level}")
        choice = client().chat.completions.create(**kw).choices[0]
        msg = choice.message
        if choice.finish_reason != "length":
            return _merge_truncated(msg, truncated), level
        print(f"[WARN] 模型输出达到 max_tokens={max_tokens} 被截断（effort={level}）")
        if (msg.content or "").strip() or msg.tool_calls or i == len(levels) - 1:
            return _merge_truncated(msg, truncated), level
        # 正文为空：把已有分析（压缩要点或思考末尾）带回，降一档、不给工具，直接要最终 JSON
        reasoning = reasoning_of(msg)
        truncated.append(f"[effort={level} 的思考，被截断]\n{reasoning}")
        note = "你刚才的思考过长被截断，没有给出最终答案。\n"
        summary = summarize_reasoning(reasoning) if TRUNCATE_SALVAGE == "summary" else ""
        if summary:
            truncated.append(f"[被截断思考的要点整理]\n{summary}")
            note += f"以下是你此前思考的要点整理：\n{summary}\n\n"
        elif TRUNCATE_SALVAGE != "none" and TRUNCATE_REASONING_TAIL > 0 and reasoning:
            note += f"以下是你思考的最后部分：\n{reasoning[-TRUNCATE_REASONING_TAIL:]}\n\n"
        msgs = messages + [{"role": "user", "content": note +
                            "不要再展开新的计算，直接根据已有分析选定着法，按系统提示输出完整 JSON。"}]
        use_tools = None
        print(f"[SALVAGE] 思考被截断（{len(reasoning)} 字），降档 {level} -> {levels[i + 1]} 直接要结论")
        live.stage(f"思考过长被截断，降档到 {levels[i + 1]}")
    return msg, level


SUMMARY_PROMPT = """下面是一段对国际象棋局面的思考过程，因为太长在中途被截断了。
请把它压缩成要点，供思考者据此直接选定着法。只整理原文已有的内容，不要补充新的分析或计算。

按下面几项列出（原文没涉及的项写"无"）：
1. 考虑过的候选着法：每个一行，写清原文对它的结论（可行 / 被否决及原因 / 未算完）
2. 发现的威胁与战术：对方的威胁、我方的战术机会、悬空或被攻击的子
3. 目前倾向的着法及理由；若原文还没倾向，写最后正在分析的着法和进展

总共不超过 400 字。

【被截断的思考】
{reasoning}"""


def summarize_reasoning(reasoning: str) -> str:
    """把被截断的思考压缩成要点（不思考的独立调用）；失败或为空时返回空串，由调用方退回带末尾。"""
    if not reasoning.strip():
        return ""
    live.stage("整理被截断思考的要点")
    try:
        resp = client().chat.completions.create(
            model=MODEL, temperature=LLM_TEMPERATURE, max_tokens=TRUNCATE_SUMMARY_MAX_TOKENS,
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(reasoning=reasoning)}],
            **effort_kwargs("off"))
        summary = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # 压缩只是补救手段，出错不能让这一步棋失败
        print(f"[SALVAGE] 要点整理失败：{e}")
        return ""
    print(f"[SALVAGE] 要点整理（{len(reasoning)} 字 -> {len(summary)} 字）：{summary[:200]}")
    return summary


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
