"""
极简 RAG 存储：jsonl 文件 + OpenAI embeddings + numpy 余弦相似度。
不依赖向量数据库，方便随时查看 / 编辑 data/*.jsonl。
"""
import json
import os
import threading
from typing import Callable, List, Dict, Any

import numpy as np


class RAGStore:
    def __init__(self, path: str, embed_fn: Callable[[str], List[float]]):
        self.path = path
        self.embed_fn = embed_fn
        self.entries: List[Dict[str, Any]] = []
        self.lock = threading.Lock()  # 复盘并行写入：查重 + 追加要原子地做
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

    def add(self, text: str, meta: Dict[str, Any] | None = None,
            dedupe_threshold: float | None = None) -> bool:
        """写入一条记录。dedupe_threshold 不为 None 时，若库中已有余弦相似度
        ≥ 阈值的条目则跳过。返回是否真的写入。"""
        emb = self.embed_fn(text)  # embedding 在锁外做，并行写入时不互相等
        with self.lock:
            if dedupe_threshold is not None and self.entries:
                q = np.array(emb, dtype=np.float32)
                qn = np.linalg.norm(q) + 1e-9
                for e in self.entries:
                    v = np.array(e["emb"], dtype=np.float32)
                    if v.shape != q.shape:
                        continue
                    if float(q @ v / (qn * (np.linalg.norm(v) + 1e-9))) >= dedupe_threshold:
                        return False
            entry = {"text": text, "meta": meta or {}, "emb": emb}
            self.entries.append(entry)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return True

    def query(self, text: str, k: int = 3, min_score: float | None = None,
              filter_fn: Callable[[Dict[str, Any]], bool] | None = None) -> List[Dict[str, Any]]:
        """余弦相似度 top-k。filter_fn 只保留满足条件的条目；min_score 过滤掉相似度过低的结果。"""
        if not self.entries:
            return []
        try:
            q = np.array(self.embed_fn(text), dtype=np.float32)
        except Exception as e:
            print(f"[RAG] embedding failed: {e}")
            return []
        qn = np.linalg.norm(q) + 1e-9
        scored = []
        skipped = 0
        for e in self.entries:
            if filter_fn is not None and not filter_fn(e):
                continue
            v = np.array(e["emb"], dtype=np.float32)
            if v.shape != q.shape:  # 换了 embedding 模型后遗留的旧向量，维度对不上
                skipped += 1
                continue
            s = float(q @ v / (qn * (np.linalg.norm(v) + 1e-9)))
            scored.append((s, e))
        if skipped and not getattr(self, "_warned_dim", False):
            print(f"[RAG] {self.path}: 跳过 {skipped} 条维度不匹配的旧向量，请运行 python reembed.py 重建")
            self._warned_dim = True
        scored.sort(key=lambda x: -x[0])
        if min_score is not None:
            scored = [(s, e) for s, e in scored if s >= min_score]
        return [{"text": e["text"], "meta": e["meta"], "score": s}
                for s, e in scored[:k]]

    def __len__(self):
        return len(self.entries)
