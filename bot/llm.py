"""LLM 调用：OpenAI 兼容客户端、思考档位阶梯、截断后的降档补救、JSON 提取。"""
import json

from openai import OpenAI

from .config import (EFFORT_LEVELS, LLM_API_KEY, LLM_BASE_URL, LLM_EXTRA_BODY, LLM_MAX_TOKENS,
                     LLM_TEMPERATURE, MODEL, THINK_LADDER, TRUNCATE_REASONING_TAIL)
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
    从 levels[0] 开始；若思考把 max_tokens 用光、正文为空，则带回思考末尾、换下一档直接要结论。"""
    levels = levels or THINK_LADDER[:1]
    max_tokens = max_tokens or LLM_MAX_TOKENS
    msgs, use_tools = messages, tools
    for i, level in enumerate(levels):
        kw = dict(model=MODEL, messages=msgs, temperature=LLM_TEMPERATURE,
                  max_tokens=max_tokens, **effort_kwargs(level))
        if use_tools:
            kw["tools"] = use_tools
        print(f"[LLM] effort={level}")
        choice = client().chat.completions.create(**kw).choices[0]
        msg = choice.message
        if choice.finish_reason != "length":
            return msg, level
        print(f"[WARN] 模型输出达到 max_tokens={max_tokens} 被截断（effort={level}）")
        if (msg.content or "").strip() or msg.tool_calls or i == len(levels) - 1:
            return msg, level
        # 正文为空：带回思考末尾，降一档、不给工具，直接要最终 JSON
        reasoning = reasoning_of(msg)
        tail = reasoning[-TRUNCATE_REASONING_TAIL:] if TRUNCATE_REASONING_TAIL > 0 else ""
        note = (f"你刚才的思考过长被截断，没有给出最终答案。以下是你思考的最后部分：\n{tail}\n\n"
                if tail else "你刚才的思考过长被截断，没有给出最终答案。\n")
        msgs = messages + [{"role": "user", "content": note +
                            "不要再展开新的计算，直接根据已有分析选定着法，按系统提示输出完整 JSON。"}]
        use_tools = None
        print(f"[SALVAGE] 思考被截断（{len(reasoning)} 字），降档 {level} -> {levels[i + 1]} 直接要结论")
        live.stage(f"思考过长被截断，降档到 {levels[i + 1]}")
    return msg, level


def reasoning_of(msg) -> str:
    """思考模型返回的推理过程（deepseek 等放在 reasoning_content），没有则为空串。"""
    return (getattr(msg, "reasoning_content", None)
            or (getattr(msg, "model_extra", None) or {}).get("reasoning_content") or "")
