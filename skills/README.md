# 技能（skill）

每个技能一个目录：`skills/<name>/SKILL.md`。技能是从过往对局总结出的、某类局面的做法与检查清单，
对局中按两种方式交给模型：

- **自动加载**：`when` 里的条件由程序按规则判定（不调 LLM、不做 embedding），命中就把正文放进这一步的局面描述。
  每步最多 `SKILL_AUTO_MAX`（默认 2）个，按 `priority`、条件数（越具体越先）排序。
- **按需读取**：所有技能的 `name：description` 常驻系统提示（技能目录），模型可调用 `load_skill(name)` 读取全文。
  没有 `when` 的技能只能这样读取。

## 格式

```markdown
---
name: stop-passed-pawns          # 与目录名相同
description: 一句话说明什么局面用、核心做法是什么（会出现在技能目录里）
priority: 3                      # 可选，默认 0，越大越优先自动加载
when:                            # 可选。一组条件，全部满足才命中
  opp_passer_min: 6
---
正文：要点、典型错误、检查清单。控制在 20 行以内，模型每步都可能读到它。
```

多组条件（任一组命中即可）用 `- ` 开头：

```yaml
when:
  - opp_captured: true
  - threatened: true
```

值可以写成 JSON（数字、`true`/`false`、`["开局", "中局"]`），列表表示"其中任一"。

## 触发条件

全部是按规则判定的局面事实，"我方"指轮到走棋的一方：

| 键 | 值 | 含义 |
|---|---|---|
| `phase` | `开局` / `中局` / `残局` | 局面阶段（与 prompt 里的阶段判定相同：前 10 回合为开局，双方轻重子总分 ≤26 为残局） |
| `min_fullmove` / `max_fullmove` | 整数 | 全回合数范围 |
| `lead_min` / `lead_max` | 整数 | 我方子力分差（兵1 马象3 车5 后9） |
| `only` | 子种列表，如 `["R", "P"]` | 盘上（不计王）只有这些子种 |
| `has` | 子种列表 | 盘上（任一方）必须有这些子种 |
| `queens` | `true` / `false` | 盘上有没有后 |
| `my_king` / `opp_king` | `uncastled` / `kingside` / `queenside` / `other` | 王的位置：还在初始格 / g、h 线己方前两排 / a-c 线己方前两排 / 其他 |
| `in_check` | `true` / `false` | 我方正被将军 |
| `opp_captured` | `true` / `false` | 对方上一步吃了子 |
| `threatened` | `true` / `false` | 我方有马象车后被更低价值的子攻击，或被攻击且没有保护 |
| `opp_passer_min` / `my_passer_min` | 整数 1-8 | 对方 / 我方最靠前的通路兵已到第几横排（按该方视角） |
| `my_color` | `white` / `black` | 我方执哪一方 |
| `opening` | SAN 前缀列表，如 `["e4 e6"]` | 对局着法以其中之一开头（纯 SAN，不带回合号） |

新增条件：在 `bot/skills.py` 的 `PREDICATES` 里加一个判定函数，并补到上表。

## 维护

- 直接编辑 SKILL.md，重启程序生效；改动用 git 审查。
- `python curate_skills.py`：把经验库里还没整理过的教训归并进现有技能（或提议新技能），写回 SKILL.md，
  之后用 `git diff skills/` 审查。`--dry-run` 只打印不写入。
- 赛后统计写在 `data/skill_stats.json`：每个技能命中的局数、步数、其中被 Stockfish 判为 blunder 的步数。
  命中很多、blunder 比例却不降的技能，说明写法没起作用，需要改写或删除。
