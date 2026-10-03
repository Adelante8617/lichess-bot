"""
换 embedding 模型后，用当前配置（.env 里的 EMBED_*）重新计算 data/*.jsonl 的向量。
文本和 meta 原样保留，只替换 emb 字段；改写前自动备份到 data/backup/。

用法：
  python reembed.py                 # 重建 openings 和 experience
  python reembed.py data/experience.jsonl
"""
import json
import os
import shutil
import sys
from datetime import datetime

import main as bot  # 复用同一个 embed()，保证与线上查询用的是同一个模型

DEFAULT_FILES = ["data/openings.jsonl", "data/experience.jsonl"]


def reembed(path: str):
    if not os.path.exists(path):
        print(f"skip (not found): {path}")
        return
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    os.makedirs("data/backup", exist_ok=True)
    name = os.path.basename(path).replace(".jsonl", "")
    bak = f"data/backup/{name}_{datetime.now():%Y%m%d_%H%M%S}_before_reembed.jsonl"
    shutil.copy2(path, bak)
    print(f"{path}: {len(rows)} 条，备份 -> {bak}")

    for i, r in enumerate(rows, 1):
        r["emb"] = bot.embed(r["text"])
        if i % 20 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    tmp = path + ".tmp"  # 先写临时文件再替换，中途失败不会损坏原文件
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    dim = len(rows[0]["emb"]) if rows else 0
    print(f"{path}: 完成，新向量维度 {dim}")


if __name__ == "__main__":
    for p in (sys.argv[1:] or DEFAULT_FILES):
        reembed(p)
