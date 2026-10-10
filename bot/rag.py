"""
极简 RAG 存储：jsonl 文件 + OpenAI embeddings + numpy 余弦相似度。
不依赖向量数据库，方便随时查看 / 编辑 data/*.jsonl。
"""
import json
import os
import threading
from typing import Callable, List, Dict, Any

import numpy as np


MAX_VARIANTS = 10


def merge_into(entry: Dict[str, Any], text: str, meta: Dict[str, Any]):
    """把一条相似记录并入 entry：seen 累加，last_seen 取较晚的 time，原文保留在 meta.variants 里。
    相似度高不等于说法一致（"别撤离防守子"和"别消极防守"向量很近），所以不丢弃被并入的原文。"""
    m = entry["meta"]
    m["seen"] = m.get("seen", 1) + meta.get("seen", 1)
    if meta.get("time"):
        m["last_seen"] = max(m.get("last_seen", ""), meta["time"])
    if text != entry["text"]:
        variants = m.setdefault("variants", [])
        for t in [text] + meta.get("variants", []):
            if t not in variants and len(variants) < MAX_VARIANTS:
                variants.append(t)


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
            if dedupe_threshold is not None and self._best_match(emb)[1] >= dedupe_threshold:
                return False
            self._append({"text": text, "meta": meta or {}, "emb": emb})
        return True

    def upsert(self, text: str, meta: Dict[str, Any] | None, threshold: float,
               filter_fn: Callable[[Dict[str, Any]], bool] | None = None) -> str:
        """写入一条记录；已有余弦相似度 ≥ threshold 的同类条目（filter_fn 筛选）时不新增，
        而是并入那一条（merge_into）。同一条教训反复出现说明它重要，而不是该存很多遍。
        返回 "added" / "merged"。"""
        emb = self.embed_fn(text)
        meta = dict(meta or {})
        with self.lock:
            hit, score = self._best_match(emb, filter_fn)
            if hit is not None and score >= threshold:
                merge_into(hit, text, meta)
                self._rewrite()
                return "merged"
            meta.setdefault("seen", 1)
            self._append({"text": text, "meta": meta, "emb": emb})
        return "added"

    def remove(self, pred: Callable[[Dict[str, Any]], bool]) -> int:
        """删除满足 pred 的条目，返回删除条数。"""
        with self.lock:
            kept = [e for e in self.entries if not pred(e)]
            removed = len(self.entries) - len(kept)
            if removed:
                self.entries = kept
                self._rewrite()
        return removed

    def save(self):
        """按内存中的 entries 整体重写文件（直接改了 entries 里的 meta 之后调用）。"""
        with self.lock:
            self._rewrite()

    def _best_match(self, emb, filter_fn=None) -> tuple[Dict[str, Any] | None, float]:
        """与 emb 余弦相似度最高的条目及分数；没有可比的条目时返回 (None, -1)。"""
        q = np.array(emb, dtype=np.float32)
        qn = np.linalg.norm(q) + 1e-9
        best, best_s = None, -1.0
        for e in self.entries:
            if filter_fn is not None and not filter_fn(e):
                continue
            v = np.array(e["emb"], dtype=np.float32)
            if v.shape != q.shape:
                continue
            s = float(q @ v / (qn * (np.linalg.norm(v) + 1e-9)))
            if s > best_s:
                best, best_s = e, s
        return best, best_s

    def _append(self, entry: Dict[str, Any]):
        self.entries.append(entry)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _rewrite(self):
        tmp = self.path + ".tmp"  # 先写临时文件再替换，中途失败不会损坏原文件
        with open(tmp, "w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

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
