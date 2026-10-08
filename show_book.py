"""查看背谱：python show_book.py [--min-n N]

每行是一个谱内局面里记下的着法：走到该局面的着法顺序 → 谱里的着法，
n = 被选中次数（入谱算 1，重新推理又选它 +1），p = 现在命中时直接照走的概率，
入谱 = 赛后评估通过而入谱的局数，评估 = 走完后 Stockfish 评分（我方视角，单位兵）的平均值。
"""
import argparse
import json
import sys

from bot import config
from bot.book import play_prob

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=1, help="只显示确认次数 n 不小于该值的着法")
    args = ap.parse_args()
    try:
        with open(config.BOOK_PATH, encoding="utf-8") as f:
            positions = json.load(f).get("positions", {})
    except FileNotFoundError:
        print(f"还没有谱（{config.BOOK_PATH} 不存在）：打完一局、赛后评估通过后才会生成。")
        return
    rows = []
    for epd, moves in positions.items():
        side = "白" if epd.split()[1] == "w" else "黑"
        for e in moves.values():
            n = 1 + e.get("confirm", 0)
            if n >= args.min_n:
                rows.append((side, e.get("via", ""), e["san"], n, e["count"], e["cp_sum"] / e["count"] / 100))
    rows.sort(key=lambda r: (r[0] != "白", len(r[1].split()), r[1], -r[3]))
    if not rows:
        print("谱是空的。")
        return
    for side in ("白", "黑"):
        part = [r for r in rows if r[0] == side]
        if not part:
            continue
        print(f"\n== 我执{side}（{len(part)} 条）==")
        for _, via, san, n, count, avg in part:
            print(f"{via or '（开局）':<40} → {san:<7} n={n:<3} p={play_prob(n):.2f}  入谱 {count} 次  评估 {avg:+.2f}")


if __name__ == "__main__":
    main()
