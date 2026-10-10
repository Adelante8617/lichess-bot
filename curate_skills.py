"""
把经验库里赛后新增的教训整理进 skills/（详见 bot/curate.py、skills/README.md）。

用法：
  python tidy_memory.py                # 建议先清理：删除被证伪的教训、合并重复
  python curate_skills.py --dry-run    # 只打印归类结果和技能改动，不写文件
  python curate_skills.py              # 写回 skills/*/SKILL.md，教训标记为已整理
  git diff skills/                     # 审查；不满意 git checkout skills/ 回退
"""
import argparse

from bot.curate import curate


def main():
    ap = argparse.ArgumentParser(description="把经验库里的教训整理进技能")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写技能文件、不标记教训")
    ap.add_argument("--batch", type=int, default=30, help="每次归类的教训条数")
    ap.add_argument("--min-new", type=int, default=3, help="新主题至少攒够几条教训才起草新技能")
    ap.add_argument("--limit", type=int, default=None, help="本次最多处理几条教训")
    args = ap.parse_args()
    curate(batch_size=args.batch, min_new=args.min_new, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
