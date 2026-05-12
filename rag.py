"""
极简 RAG 存储：jsonl 文件 + OpenAI embeddings + numpy 余弦相似度。
不依赖向量数据库，方便随时查看 / 编辑 data/*.jsonl。
"""
import json
import os
from typing import Callable, List, Dict, Any

import numpy as np


class RAGStore:
    def __init__(self, path: str, embed_fn: Callable[[str], List[float]]):
        self.path = path
        self.embed_fn = embed_fn
        self.entries: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.entries.append(json.loads(line))
                except Exception:
                    pass

    def add(self, text: str, meta: Dict[str, Any] | None = None):
        emb = self.embed_fn(text)
        entry = {"text": text, "meta": meta or {}, "emb": emb}
        self.entries.append(entry)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def query(self, text: str, k: int = 3) -> List[Dict[str, Any]]:
        if not self.entries:
            return []
        try:
            q = np.array(self.embed_fn(text), dtype=np.float32)
        except Exception as e:
            print(f"[RAG] embedding failed: {e}")
            return []
        qn = np.linalg.norm(q) + 1e-9
        scored = []
        for e in self.entries:
            v = np.array(e["emb"], dtype=np.float32)
            s = float(q @ v / (qn * (np.linalg.norm(v) + 1e-9)))
            scored.append((s, e))
        scored.sort(key=lambda x: -x[0])
        return [{"text": e["text"], "meta": e["meta"], "score": s}
                for s, e in scored[:k]]

    def __len__(self):
        return len(self.entries)
