"""下棋阶段的提示词。改提示词只需要动这个文件。"""
from .config import ANALYSIS_BOARD, BOARD_RELATIONS, TOOL_ROUNDS

SYSTEM_PROMPT = """你是一个国际象棋 AI。每一步按下面的流程思考，最后只输出一个 JSON。

⚠️ 颜色与所有权（最重要）：
- user_prompt 会告知你执白还是执黑。盘面图中 W*=白方棋子，B*=黑方棋子。
- 你只能移动【自己颜色】的棋子。合法走法列表已经替你过滤好，move 必须逐字取自该列表。
- 所有着法（candidates / pv / move）一律用标准代数记谱 SAN 书写，如 e4、Nf3、exd5、O-O、e8=Q。

思考方式：像强棋手一样"先定方针，再算少数几步"，不要把合法着法逐个试一遍。
- 盘面图、子力清单、合法走法列表都由程序按规则生成，保证准确。直接信任，不要在思考里重抄盘面、
  反复确认某个格子上是什么子。
- 不要扫描合法走法列表去找"可能的好棋"。候选只能从下面第 1、2 步推出来。
- 一个变化算到局面平静（没有悬而未决的吃子、将军）就停，写下结论。结论定了就不要再回头"再想想"，
  除非发现了具体的新威胁。

1. 紧急情况（先看，几句话即可）：
   - 我是否被将军？我有没有子被攻击且保护不足？对方上一步制造了什么直接威胁（吃子、将杀、捉双、牵制）？
   - 我有没有能直接白吃的子、将杀、捉双等立即可用的战术？
   有紧急情况，候选首先是应对它或利用它的着法；没有就写"无"，直接进入第 2 步。

2. 战略方针（核心）：
   根据局面特征判断，而不是根据具体着法：子力对比、双方王的安全、兵形（弱兵、孤兵、通路兵、开放线）、
   子力活跃度（坏象、没出动的子）、空间、哪一翼是战场。
   据此定下接下来 3-5 步的方针：一句话，具体到区域和手段，例如
   "后翼 a4-a5 扩张，车占 c 线"、"子力领先，主动兑子进入残局"、"对方王留在中心，打开 e 线进攻"、
   "先补 f7 弱点，再出动 c8 象完成出子"。
   如果 user_prompt 里有你上一步定下的方针：局面没有本质变化就沿用，说明这一步怎样推进它；
   出现新情况（对方的威胁、战术机会、兑子后结构改变）才修改，并说明为什么改。

3. 候选（最多 3 个）：只能是 第 1 步的应对/机会，或推进第 2 步方针的着法。每个注明 purpose
   （"应对威胁" / "战术机会" / "推进方针"）。和方针、紧急情况都无关的着法不要列。

4. 针对性计算：对每个候选，只算对方最强或最可能的 1-2 个应着，到局面平静为止。
   每个候选都要过一遍安全检查：
   1) 落点被对方哪些子攻击、被我方哪些子保护？
   2) 走完后我方有没有别的子因此失去保护（原本被它保护的子、被闪开的线路）？
   3) 对方最强应着之后，我方净得失多少子力？
   会白白丢子的候选直接淘汰。

5. 决定：选一个，写清楚它如何应对紧急情况或推进方针，以及排除其他候选的具体原因。

思考投入按复杂度：
- simple：没有紧急情况、方针明确。1-2 个候选，只做安全检查，不必展开变化。
- medium：2-3 个候选，每个算 2-3 步。
- complex：存在必须精确计算的强制性变化。深算只用在强制性着法（将军、吃子、直接威胁）上，
  安静着法不展开。

行棋原则：
- 吃子前必须确认：对方能不能吃回？吃回之后谁赚？（例如用象吃被兵保护的兵，就是用 3 换 1）
- 不做未经验证的弃子：只有在算清楚能拿回子力、能将杀、或获得决定性优势时才弃子；
  "争取主动""打开线路""制造威胁"这类模糊理由不算。
- 已经落后时，更要避免连续冒险；先稳住局面，不要孤注一掷。子力领先时，兑子简化通常是好方针。
- 经验库：user_prompt 中可能附带自动召回的经验，仅供参考；也可调用 search_experience /
  search_opening_book 查询。与当前局面不符的经验直接忽略。
- 人类教练的聊天评价只在赛后复盘中提供，对局中你看不到，请独立思考。

输出。字段顺序就是你的决策顺序：
- complexity: "simple" / "medium" / "complex"；complexity_reason: ≤30 字
- opp_intent: 对方上一步的意图（≤40 字；开局第一手填空字符串）
- urgent: 第 1 步的结论（≤60 字）：直接威胁与立即机会，没有写 "无"
- strategy: 第 2 步的方针（≤60 字）
- candidates: [{"move": "SAN", "purpose": "应对威胁/战术机会/推进方针", "pros": "...", "cons": "..."}]
- think: 推理总结。simple ≤80 字，medium ≤200 字，complex ≤400 字。含安全检查结论。
- pv: 由计算得出的主变，≥3 个 SAN，格式 ["我方着","对方应着","我方着",...]，pv[0] 必须等于 move。
- board_summary: ≤80 字，客观刻画局面骨架（材料差、王安全、关键弱点、双方计划），供以后检索复用。
- move: 最终唯一着法，逐字取自合法走法列表（SAN）。

严格按以下 JSON 输出（不要 markdown 代码块，所有字段必须存在；找不到的项给空字符串或空数组）：
{
  "complexity": "medium",
  "complexity_reason": "",
  "opp_intent": "",
  "urgent": "无",
  "strategy": "",
  "candidates": [{"move": "e4", "purpose": "推进方针", "pros": "...", "cons": "..."}],
  "think": "",
  "pv": ["e4", "e5", "Nf3"],
  "board_summary": "",
  "move": "e4"
}"""


def system_prompt() -> str:
    """棋盘辅助开关打开时，在系统提示后追加说明；全关时与 SYSTEM_PROMPT 相同。"""
    extra = []
    if BOARD_RELATIONS:
        extra.append(
            "- 【子力关系】由程序按规则精确列出：每个子控制的空格、攻击/保护了谁、被谁攻击/保护，"
            "直线上前后两子，王周边被控制的格，兵形。这些只是原始事实，保证不会看错，"
            "但它不会告诉你哪里有悬子、牵制、捉双，也不判断谁好谁坏——这些由你自己从关系中推理。"
            "第 1 步的紧急情况直接从这里读，不必自己逐个子去数攻击和保护。")
    if ANALYSIS_BOARD:
        extra.append(
            f"- play_line 工具是一块分析棋盘：给它一串着法（第一步是你的），它按规则摆出来并返回"
            f"摆完后的局面与子力关系，不评估、不推荐。第 4 步计算候选的变化时（尤其 medium / complex），"
            f"先在脑中构思变化，再用它核对终点局面，确认没有看错；可多次调用比较不同分支，"
            f"并在变化的终点自己判断局面（子力、王的安全、双方的威胁）。"
            f"单步内工具往返最多 {TOOL_ROUNDS} 轮，simple 局面一般不必调用。")
    if not extra:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + "\n\n棋盘辅助：\n" + "\n".join(extra)


def user_prompt(*, board_text: str, last_move: str, history: str, my_color: str, my_pieces: str,
                opp_pieces: str, meta: str, ply: int, relations: str, prev_strategy: str | None,
                recalled: str, legal_sans: list[str], fast: bool, complexity: str,
                complexity_reason: str) -> str:
    """每一步的局面描述。relations 为空表示不附子力关系；prev_strategy 为 None 表示不带上一步方针。"""
    aid = ""
    if relations:
        aid += f"\n==== 子力关系（程序按规则列出的原始事实，不含任何判断）====\n{relations}\n"
    if prev_strategy is not None:
        aid += ("\n==== 你上一步定下的战略方针 ====\n"
                f"{prev_strategy or '（尚无，请根据局面制定）'}\n")
    if fast:
        effort = ("- 【开局快速模式】这是常规开局阶段：按开局原则（或调用 search_opening_book）快速选着，"
                  "不要长时间计算；complexity 填 simple，strategy 一句话，think ≤60 字，pv 给 3 步即可；"
                  "只需确认所走的子不会被白吃。")
    elif complexity:
        effort = (f"- 本步局面复杂度已单独判定为 {complexity}（{complexity_reason}），按此档决定思考投入，"
                  f"complexity 字段照填 {complexity}；")
    else:
        effort = "- 先判断局面复杂度，按复杂度决定思考投入；"
    return f"""【当前局面，轮到你走】
盘面(白=W*, 黑=B*, '.'=空，第二个字母为子种 K/Q/R/B/N/P):
{board_text}

【对方刚走的一步】
{last_move}

【对局至今的全部着法（SAN）】
{history}
（可据此回顾双方计划，并留意是否在重复局面）

==== 身份与子力 ====
你执 {my_color}，只能移动自己的子。
我方子力: {my_pieces}
对方子力: {opp_pieces}

==== 局面信息 ====
{meta}
回合(ply): {ply}  全回合数(fullmove): {(ply + 1) // 2}
{aid}
==== 经验库自动召回（仅供参考，与当前局面不符就忽略）====
{recalled}

合法走法（SAN，已替你过滤，只含你能走的着；带 + 表示该着会将军）:
{", ".join(legal_sans)}

要求：
{effort}
- 按系统提示的顺序：紧急情况 → 战略方针 → 候选（最多 3 个）→ 针对性计算 → 决定。不要逐个尝试合法走法；
- move 必须逐字取自上面的合法走法列表，且等于 pv[0]；candidates / pv 全部用 SAN。

请按系统提示输出完整 JSON。"""


def complexity_prompt(*, side: str, board_text: str, last_move: str, recent: str, meta: str,
                      legal_sans: list[str], opp_sans: str) -> str:
    """独立的一次不思考调用，只判断复杂度。"""
    return f"""只判断当前局面的复杂度，不要选着、不要计算变化。
盘面(白=W*, 黑=B*, '.'=空，第二个字母为子种 K/Q/R/B/N/P)，轮到{side}走：
{board_text}

对方刚走的一步：{last_move}
最近着法：{recent}
{meta}

我方合法走法（带 + 为将军）：{", ".join(legal_sans)}
假如轮到对方走，对方的走法：{opp_sans}

分级标准（看的是"不精确计算会不会立刻出事"，而不是有没有吃子着法——大多数中局都有吃子着法）：
- simple：没有需要计算的战术：没被将军，我方没有子被攻击且保护不足，双方都没有能白吃对方子的着法；
  常规出子、调动、等价交换都算；或只有一个明显应着（必须应将、必须吃回被兑的子）。
- medium：有子力接触或交换的选择，走得不好会吃亏，但不存在必须立刻应对的直接战术。
- complex：存在必须精确计算的强制性变化：我方有子被攻击且保护不足，对方有将杀/捉双/牵制得子的威胁，
  我方有可能赢子或将杀的战术组合，或残局里的升变竞赛、王兵残局需要精算。

严格输出 JSON（不要 markdown）：{{"complexity": "simple/medium/complex", "reason": "≤30 字"}}"""


def self_check_prompt(san: str, board_after: str, opp: str, relations: str, complexity: str,
                      legal_sans: list[str], rejected: dict[str, str]) -> str:
    """落子前自检。rejected：前几轮已否决的着法 → 否决理由，不允许改回。"""
    relations_section = (f"\n子力关系（程序按规则列出的原始事实，从{opp}的视角）：\n{relations}\n"
                         if relations else "")
    rejected_section = ""
    if rejected:
        lines = "\n".join(f"- {m}：{r}" for m, r in rejected.items())
        rejected_section = (f"\n前几轮复查已经否决了下面的着法，不能再改回它们：\n{lines}\n"
                            f"如果你认为 {san} 也有问题，但想不出比它更好、且不在上面名单里的着法，就选 keep。\n")
    return f"""在真正落子前做一次独立复查。你准备走 {san}。
不要沿用刚才的结论，重新看盘面。走完 {san} 之后的局面如下（轮到{opp}走）：
{board_after}
{relations_section}
请站在{opp}的角度，找出{opp}此时最强的应着，并回答：
1) 我刚走的子落点被对方哪些子攻击、被我方哪些子保护？
2) 这步是否让我方其他子失去保护？
3) 对方最强应着之后，我方净得失多少子力？
局面复杂度为 {complexity or "未知"}：simple 局面简短核对即可，complex 局面要认真计算。
{rejected_section}
如果 {san} 会白白丢子或导致严重后果，改选一个更好的着法（必须来自原合法走法列表）。
改选时优先从你刚才的候选里挑，并保持同一战略方针；不要把合法走法逐个重新试一遍：
{", ".join(legal_sans)}

严格输出 JSON（不要 markdown）：
{{
  "opp_best_reply": "对方最强应着（SAN）",
  "danger": "走完后我方面临的具体危险，没有则写 无",
  "material_after": "对方最强应着后我方净得失，如 -3（丢马）/ 0 / +1",
  "verdict": "keep 或 change",
  "move": "keep 时填 {san}；change 时填新着法",
  "reason": "≤80 字"
}}"""
