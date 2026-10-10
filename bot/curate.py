"""把经验库里赛后新增的零散教训整理进技能（curate_skills.py 调用）。

1. 归类：分批把还没整理过的教训交给模型，归到现有技能、提议新技能（new:<名字>），或丢弃（太笼统、
   与系统提示里的行棋原则重复、只针对某一盘的具体着法）；
   各批并行、互相看不到对方起的新名字，归类完再用一次调用合并同义的新名字（也可并入现有技能）；
2. 改写：每个分到教训的技能，连同它的赛后统计，让模型在原文基础上改写（合并重复、补充新做法、删掉被证伪的），
   正文有长度上限；
3. 新建：同一个新名字下攒够 min_new 条教训才起草新技能，不够的留到下次；
4. 校验：输出必须能被 skills.parse_skill 解析、name 不变 / 合法、正文不超长，否则这个技能本次不改；
5. 已归类的教训在 meta.curated 记下去向（技能名 / discard），下次不再处理。

技能文件在 git 里，改完用 git diff skills/ 审查，不满意就 git checkout 回去。
"""
import difflib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import skills
from .config import MODEL, REVIEW_WORKERS, SKILL_STATS_PATH
from .llm import complete, extract_json
from .memory import BAD_VERDICTS, experience_rag, is_lesson, refuted_lessons

BODY_LIMIT = 1200  # 正文字数上限：模型每步都可能读到它
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def pending_lessons(entries: list[dict]) -> list[dict]:
    """还没整理过、也没被证伪的教训。"""
    refuted = {id(e) for e in refuted_lessons(entries)}
    return [e for e in entries if is_lesson(e) and not e["meta"].get("curated") and id(e) not in refuted
            and e["meta"].get("alt_verdict") not in BAD_VERDICTS]


def _clean(text: str) -> str:
    return re.sub(r"\s*\(局面: .*\)$", "", text)  # blunder 教训末尾的出处对技能没有用


def _lesson_text(e: dict) -> str:
    seen = e["meta"].get("seen", 1)
    extra = "".join(f"\n   另一种说法：{_clean(v)}" for v in e["meta"].get("variants", [])[:2])
    return _clean(e["text"]) + (f"（出现 {seen} 次）" if seen > 1 else "") + extra


def _ask(prompt: str) -> str:
    resp = complete(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3)
    return (resp.choices[0].message.content or "").strip()


def classify(batch: list[dict], index: str) -> dict[int, str]:
    """一批教训 → {批内序号: 技能名 / new:<名字> / discard}。解析失败的序号不出现在结果里。"""
    numbered = "\n".join(f"{i}. {_lesson_text(e)}" for i, e in enumerate(batch, 1))
    prompt = f"""下面是国际象棋 AI 在赛后复盘中写下的教训，需要归档到"技能"里。现有技能（name：说明）：
{index or '（还没有技能）'}

教训：
{numbered}

逐条决定去向：
- 属于某个现有技能的局面类型：填该技能的 name；
- 是现有技能都没覆盖、但会反复出现的一类局面：填 "new:<英文小写短横线名字>"，同一类的教训用同一个名字；
- 太笼统（如"保持警觉"）、与基本原则重复（不要白丢子、吃子前算交换）、或只针对某一盘的具体着法而无法推广：填 "discard"。

严格输出 JSON（不要 markdown）：{{"assign": [{{"i": 1, "to": "技能名 / new:名字 / discard"}}]}}"""
    obj = extract_json(_ask(prompt)) or {}
    out: dict[int, str] = {}
    for item in obj.get("assign") or []:
        try:
            i, to = int(item["i"]), str(item["to"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= i <= len(batch) and to:
            out[i] = to
    return out


def _strip_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def _predicate_doc() -> str:
    return "\n".join(f"- {k}：{v}" for k, v in skills.PREDICATE_DOCS.items())


FORMAT_NOTE = f"""SKILL.md 格式：
---
name: 英文小写短横线名字
description: 一句话：什么局面用、核心做法（会出现在技能目录里）
priority: 0（可选，越大越优先自动加载）
when:（可选，程序按规则判定，全部满足才自动加载；多组任一满足时每组以 "- " 开头）
  条件名: 值
---
正文：要点、典型错误、检查清单，编号列表，不超过 {BODY_LIMIT} 字。

可用的 when 条件（只能用这些）：
{{predicates}}"""


def _validate(text: str, expect_name: str | None) -> dict:
    skill = skills.parse_skill(text)
    if expect_name and skill["name"] != expect_name:
        raise ValueError(f"name 被改成了 {skill['name']}")
    if not NAME_RE.match(skill["name"]):
        raise ValueError(f"name 不合法：{skill['name']}")
    if len(skill["body"]) > BODY_LIMIT:
        raise ValueError(f"正文 {len(skill['body'])} 字，超过 {BODY_LIMIT}")
    return skill


def rewrite_skill(skill: dict, lessons: list[dict], stats: dict | None) -> str:
    """把新教训并进现有技能，返回新的 SKILL.md 全文（已校验）。"""
    with open(skill["path"], encoding="utf-8") as f:
        current = f.read()
    stat_line = (f"赛后统计：命中 {stats['games']} 局 {stats['moves']} 步，其中 {stats['blunders']} 步被 Stockfish 判为 blunder。"
                 if stats else "赛后统计：暂无。")
    items = "\n".join(f"- {_lesson_text(e)}" for e in lessons)
    prompt = f"""你在维护国际象棋 AI 的一个技能文件。下面是它现在的内容：

{current}

{stat_line}

赛后复盘新写下的、归到这个技能的教训：
{items}

请改写这个技能：
- 新教训里有用的做法补充进去；与已有条目重复的合并，不要越写越长；
- 互相矛盾时，以更具体、可检验的说法为准；
- 只写能推广到同类局面的做法，不写某一盘的具体着法（典型例子可以保留，如"h6 兵控制 g5"）；
- name 不变；when 一般不改，确实需要时只能用下面列出的条件；
- 正文不超过 {BODY_LIMIT} 字。

{FORMAT_NOTE.format(predicates=_predicate_doc())}

只输出完整的 SKILL.md 内容（从第一行 --- 开始），不要任何解释。"""
    text = _strip_fence(_ask(prompt))
    _validate(text, skill["name"])
    return text


def draft_skill(name: str, lessons: list[dict], index: str) -> str:
    """为一组教训起草新技能，返回 SKILL.md 全文（已校验）。"""
    items = "\n".join(f"- {_lesson_text(e)}" for e in lessons)
    prompt = f"""你在为国际象棋 AI 新建一个技能文件（建议名字 {name}）。现有技能（不要和它们重复）：
{index or '（还没有技能）'}

归到这个新技能的教训：
{items}

{FORMAT_NOTE.format(predicates=_predicate_doc())}

when 要尽量准确地描述这类局面（命中太宽会在无关局面里占用篇幅）；规则无法描述的就不写 when，只能由模型主动读取。
只输出完整的 SKILL.md 内容（从第一行 --- 开始），不要任何解释。"""
    text = _strip_fence(_ask(prompt))
    _validate(text, None)
    return text


class Progress:
    """终端进度条（不依赖 tqdm），线程安全：  [归类] ██████░░░░░░░░░░░░ 3/8  37%
    输出不是终端（重定向到文件）时每完成一项打印一行，避免 \\r 刷出一堆乱码。"""

    def __init__(self, label: str, total: int, width: int = 24):
        self.label, self.total, self.width = label, total, width
        self.done = 0
        self.lock = threading.Lock()
        self.tty = sys.stdout.isatty()
        self._render()

    def step(self):
        with self.lock:
            self.done += 1
            self._render()

    def _render(self):
        if self.total <= 0:
            return
        filled = self.width * self.done // self.total
        line = (f"[{self.label}] {'█' * filled}{'░' * (self.width - filled)} "
                f"{self.done}/{self.total} {100 * self.done // self.total:3d}%")
        if self.tty:
            sys.stdout.write("\r" + line + ("\n" if self.done >= self.total else ""))
            sys.stdout.flush()
        elif self.done:
            print(line)


def run_parallel(fn, items: list, label: str) -> list:
    """用 REVIEW_WORKERS 个线程对每项执行 fn，显示进度，按原顺序返回结果。
    单项抛异常时结果为 None（调用方按"没有结果"处理），不影响其他项。"""
    results: list = [None] * len(items)
    if not items:
        return results
    progress = Progress(label, len(items))
    with ThreadPoolExecutor(max_workers=max(1, REVIEW_WORKERS)) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            try:
                results[futures[fut]] = fut.result()
            except Exception as e:
                print(f"\n[CURATE] {label} 第 {futures[fut] + 1} 项失败：{e}")
            progress.step()
    return results


def merge_new_names(groups: dict[str, list[dict]], current: dict, index: str) -> dict[str, str]:
    """各批归类时独立起的新主题名，让模型一次性合并同义的（也可以并入现有技能）。
    原地修改 groups，返回 {旧名: 新去向}。少于 2 个新名字、或模型回答无效时不改。"""
    new_names = sorted(k for k in groups if k.startswith("new:"))
    if len(new_names) < 2:
        return {}
    listing = "\n".join(f"- {n}（{len(groups[n])} 条）：" + "；".join(_clean(e["text"])[:50] for e in groups[n][:3])
                        for n in new_names)
    prompt = f"""下面是把国际象棋教训分批归类时，各批分别提议的新主题名字（每个附最多 3 条教训示例）。
因为各批互相看不到，同一主题可能起了不同的名字。

新主题名字：
{listing}

现有技能（name：说明）：
{index or '（还没有技能）'}

请合并同义或高度重叠的新主题：把每个需要改名的新名字映射到另一个新名字（保留的那个），
或者映射到某个现有技能的 name（这个主题其实已被现有技能覆盖）。不需要改的不要列出。

严格输出 JSON（不要 markdown）：{{"merge": {{"new:旧名字": "new:保留的名字 或 现有技能名"}}}}"""
    print(f"[CURATE] 合并新主题名字（{len(new_names)} 个）...")
    try:
        obj = extract_json(_ask(prompt)) or {}
    except Exception as e:
        print(f"[CURATE] 合并新主题名字失败：{e}")
        return {}
    raw = obj.get("merge") if isinstance(obj.get("merge"), dict) else {}
    valid = set(new_names) | set(current)
    mapping = {str(k).strip(): str(v).strip() for k, v in raw.items()
               if str(k).strip() in new_names and str(v).strip() in valid and str(k).strip() != str(v).strip()}

    def resolve(name: str) -> str:  # 处理 a→b、b→c 这样的链，防止成环
        seen = {name}
        while name in mapping and mapping[name] not in seen:
            name = mapping[name]
            seen.add(name)
        return name

    renamed = {}
    for old in list(mapping):
        target = resolve(old)
        if target == old or old not in groups:
            continue
        groups.setdefault(target, []).extend(groups.pop(old))
        renamed[old] = target
    if renamed:
        print("[CURATE] 合并：" + "，".join(f"{a} → {b}" for a, b in renamed.items()))
    return renamed


def _load_stats() -> dict:
    try:
        with open(SKILL_STATS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def curate(root: str | None = None, batch_size: int = 30, min_new: int = 3, limit: int | None = None,
           dry_run: bool = False, store=None) -> dict:
    """完整流程，返回 {"changed": {技能名: 新全文}, "created": {...}, "assigned": {去向: 条数}, "failed": [...]}。"""
    root = root or skills.SKILLS_DIR
    store = store or experience_rag
    current = skills.load_skills(root)
    index = skills.index_text(current)
    lessons = pending_lessons(store.entries)[:limit]
    print(f"[CURATE] 待整理教训 {len(lessons)} 条，现有技能 {len(current)} 个")

    batches = [lessons[i:i + batch_size] for i in range(0, len(lessons), batch_size)]
    results = run_parallel(lambda b: classify(b, index), batches, "归类")
    groups: dict[str, list[dict]] = {}
    for batch, assign in zip(batches, results):
        for i, to in (assign or {}).items():
            if to != "discard" and not to.startswith("new:") and to not in current:
                continue  # 编出来的技能名：不归档，下次再分
            groups.setdefault(to, []).append(batch[i - 1])
    # 各批独立起名，同一主题可能叫 new:queen-safety / new:protect-queen：统一合并一次
    renamed = merge_new_names(groups, current, index)
    report = {"changed": {}, "created": {}, "failed": [], "renamed": renamed,
              "assigned": {k: len(v) for k, v in sorted(groups.items())}}
    print("[CURATE] 归类：" + "，".join(f"{k}×{n}" for k, n in report["assigned"].items()))

    stats = _load_stats()
    jobs = [("rewrite", name, items) for name, items in groups.items() if name in current]
    jobs += [("draft", name[4:], items) for name, items in groups.items()
             if name.startswith("new:") and len(items) >= min_new and NAME_RE.match(name[4:])
             and name[4:] not in current]

    def run(job):
        kind, name, items = job
        try:
            if kind == "rewrite":
                return job, rewrite_skill(current[name], items, stats.get(name)), None
            return job, draft_skill(name, items, index), None
        except Exception as e:  # 单个技能失败不影响其他技能
            return job, None, e

    outputs = run_parallel(run, jobs, "改写 / 新建技能")

    done: dict[int, str] = {}  # id(教训) → 去向
    for (kind, name, items), text, err in outputs:
        if err is not None:
            print(f"[CURATE] {name} 失败：{err}")
            report["failed"].append(name)
            continue
        if kind == "rewrite":
            with open(current[name]["path"], encoding="utf-8") as f:
                old = f.read()
            report["changed"][name] = text
            print("".join(difflib.unified_diff(old.splitlines(True), text.splitlines(True),
                                               f"a/{name}/SKILL.md", f"b/{name}/SKILL.md")))
            path = current[name]["path"]
        else:
            name = skills.parse_skill(text)["name"]
            if name in current or name in report["created"]:
                print(f"[CURATE] 新技能名 {name} 与已有技能重名，跳过")
                report["failed"].append(name)
                continue
            report["created"][name] = text
            print(f"[CURATE] 新技能 {name}：\n{text}")
            path = os.path.join(root, name, "SKILL.md")
        if not dry_run:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        for e in items:
            done[id(e)] = name
    for e in groups.get("discard", []):
        done[id(e)] = "discard"

    if not dry_run and done:
        for e in store.entries:
            if id(e) in done:
                e["meta"]["curated"] = done[id(e)]
        store.save()
        skills.reload()
    left = sum(1 for e in lessons if id(e) not in done)
    print(f"[CURATE] 改写 {len(report['changed'])} 个，新建 {len(report['created'])} 个，"
          f"丢弃 {len(groups.get('discard', []))} 条，留到下次的教训 {left} 条"
          + ("（dry-run，未写入）" if dry_run else ""))
    return report
