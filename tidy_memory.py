"""
整理经验库 data/experience.jsonl（不调用 LLM，也不重新计算 embedding）：
1. 删除被证伪的 blunder 教训：同一个 blunder 里模型给出的替代着法被 Stockfish 判为 mistake / blunder；
2. 合并重复教训：同类教训相似度 ≥ 阈值（默认 LESSON_MERGE_SIM）的只留最早一条，meta.seen 累加。
改写前自动备份到 data/backup/。新写入的教训在写入时就会合并，这个脚本用来清理旧库。

用法：
  python tidy_memory.py --dry-run          # 只列出会删除 / 合并的条目
  python tidy_memory.py                    # 执行
  python tidy_memory.py --threshold 0.92
"""
import argparse
import os
import shutil
from datetime import datetime

from bot.config import LESSON_MERGE_SIM
from bot.memory import consolidate, experience_rag


def main():
    ap = argparse.ArgumentParser(description="整理经验库：删除被证伪的教训，合并重复教训")
    ap.add_argument("--threshold", type=float, default=LESSON_MERGE_SIM, help="合并阈值（余弦相似度）")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不改文件")
    args = ap.parse_args()

    path = experience_rag.path
    before = len(experience_rag)
    if not args.dry_run and os.path.exists(path):
        os.makedirs("data/backup", exist_ok=True)
        bak = f"data/backup/experience_{datetime.now():%Y%m%d_%H%M%S}_before_tidy.jsonl"
        shutil.copy2(path, bak)
        print(f"备份 -> {bak}")

    report = consolidate(experience_rag, args.threshold, dry_run=args.dry_run)
    print(f"\n== 被证伪的 blunder 教训：{len(report['refuted'])} 条 ==")
    for e in report["refuted"]:
        print(f"- {e['text'][:100]}")
    print(f"\n== 合并的重复教训：{len(report['merged'])} 条（阈值 {args.threshold}）==")
    for keep, dup in report["merged"]:
        print(f"- 保留：{keep['text'][:80]}\n  并入：{dup['text'][:80]}")
    removed = len(report["refuted"]) + len(report["merged"])
    if args.dry_run:
        print(f"\n[dry-run] {before} 条，将删除 {removed} 条")
    else:
        print(f"\n{before} 条 -> {len(experience_rag)} 条")


if __name__ == "__main__":
    main()
