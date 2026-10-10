"""RAG：开局库 + 经验记忆库，以及每步的经验自动召回。"""
import re
import threading

import chess
import numpy as np
from openai import OpenAI

from .boardtext import COLOR_ZH, game_phase, material_text, san_history
from .config import (AUTO_RECALL_K, EMBED_API_KEY, EMBED_BACKEND, EMBED_BASE_URL, EMBED_DEVICE,
                     EMBED_LOCAL_MODEL, EMBED_MODEL, EMBED_QUERY_PREFIX, LESSON_MERGE_SIM)
from .rag import RAGStore, merge_into

_emb_client = None
_local_embedder = None
_init_lock = threading.Lock()  # 复盘并行时多个线程可能同时第一次调用 embed


def embed(text: str):
    global _emb_client, _local_embedder
    if EMBED_BACKEND == "local":
        with _init_lock:
            if _local_embedder is None:
                from sentence_transformers import SentenceTransformer
                print(f"[EMBED] loading local model {EMBED_LOCAL_MODEL} (device={EMBED_DEVICE or 'auto'}) ...")
                _local_embedder = SentenceTransformer(EMBED_LOCAL_MODEL, device=EMBED_DEVICE)
        return _local_embedder.encode(text, normalize_embeddings=True).tolist()
    with _init_lock:
        if _emb_client is None:
            _emb_client = OpenAI(api_key=EMBED_API_KEY, base_url=EMBED_BASE_URL)
    resp = _emb_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


opening_rag = RAGStore("data/openings.jsonl", embed)
experience_rag = RAGStore("data/experience.jsonl", embed)


def warm_up():
    """后台预热 embedding（本地模型首次加载要十几秒）：启动时就加载，不让第一步召回卡住对局。"""
    def _run():
        try:
            embed("warm up")
        except Exception as e:
            print(f"[EMBED] warm-up failed: {e}")

    threading.Thread(target=_run, daemon=True).start()


def seed_openings_if_empty():
    """首次启动时给开局库塞一些基础开局思路，之后通过复盘自然增长。"""
    if len(opening_rag) > 0:
        return
    seeds = [
        ("Italian Game: 1.e4 e5 2.Nf3 Nc6 3.Bc4。快速出动轻子，向 f7 施压。",
         {"name": "Italian Game"}),
        ("Ruy Lopez: 1.e4 e5 2.Nf3 Nc6 3.Bb5。主教钉马形成长期压力。",
         {"name": "Ruy Lopez"}),
        ("Sicilian Defense: 1.e4 c5。黑方不对称反击，复杂战斗。",
         {"name": "Sicilian"}),
        ("French Defense: 1.e4 e6 2.d4 d5。黑方兵链稳固，注意 c8 象。",
         {"name": "French"}),
        ("Caro-Kann: 1.e4 c6 2.d4 d5。稳健，残局兵形好。",
         {"name": "Caro-Kann"}),
        ("Queen's Gambit: 1.d4 d5 2.c4。弃兵抢中心。",
         {"name": "Queen's Gambit"}),
        ("King's Indian Defense: 1.d4 Nf6 2.c4 g6。黑方让中心后反击。",
         {"name": "KID"}),
        ("English Opening: 1.c4。侧翼控 d5，灵活转位。",
         {"name": "English"}),
        ("London System: 1.d4 2.Nf3 3.Bf4。白方稳健体系，易掌握。",
         {"name": "London"}),
        ("Scandinavian: 1.e4 d5。黑方立刻挑战中心。",
         {"name": "Scandinavian"}),
        ("通用开局原则：抢中心、快速出动轻子、王车易位、不要早出皇后、不要重复走同一子。",
         {"name": "principles"}),
    ]
    print(f"[RAG] seeding {len(seeds)} openings ...")
    for text, meta in seeds:
        try:
            opening_rag.add(text, meta)
        except Exception as e:
            print(f"[RAG] seed failed: {e}")
            break


def search(store: RAGStore, query: str, k: int = 3) -> str:
    """供工具调用：检索并格式化成文本。"""
    hits = store.query(EMBED_QUERY_PREFIX + query, k=k)
    if not hits:
        return "（暂无相关记录）"
    return "\n".join(f"- [{h['score']:.2f}] {h['text']}" for h in hits)


def is_lesson(entry: dict) -> bool:
    """自动召回只取"教训"类条目：不含整局总结、局面快照、替代走法评估、聊天原文。"""
    return (entry["meta"].get("kind") in (None, "blunder", "chat_guidance")
            and not entry["text"].startswith(("[复盘]", "[Chat-Summary]")))


def add_lesson(text: str, meta: dict) -> str:
    """写入一条教训：已有几乎相同的同类教训时合并计数（seen +1），不重复追加。返回 "added" / "merged"。"""
    kind = meta.get("kind")
    return experience_rag.upsert(text, meta, LESSON_MERGE_SIM,
                                 filter_fn=lambda e: is_lesson(e) and e["meta"].get("kind") == kind)


BAD_VERDICTS = ("mistake", "blunder")
_ALT_RE = re.compile(r"\(Δ=-?\d+, (\w+)\)")


def alt_verdict(entry: dict) -> str:
    """[Blunder-AltMove] 条目里 Stockfish 对模型替代着法的结论（ok / inaccuracy / mistake / blunder）。
    新条目记在 meta 里，旧条目只写在文本中，从文本解析。"""
    if entry["meta"].get("alt_verdict"):
        return entry["meta"]["alt_verdict"]
    m = _ALT_RE.search(entry["text"])
    return m.group(1) if m else ""


def refuted_lessons(entries: list[dict]) -> list[dict]:
    """被证伪的 blunder 教训：同一个 blunder（time + ply 相同）里模型给出的替代着法被 Stockfish 判为
    mistake / blunder。新条目的结论记在 meta.alt_verdict，旧条目要从对应的 [Blunder-AltMove] 文本里找。"""
    refuted_keys = {(e["meta"].get("time"), e["meta"].get("ply")) for e in entries
                    if e["meta"].get("kind") == "blunder_alt_eval" and alt_verdict(e) in BAD_VERDICTS}
    return [e for e in entries if e["meta"].get("kind") == "blunder"
            and (e["meta"].get("alt_verdict") in BAD_VERDICTS
                 or (e["meta"].get("time"), e["meta"].get("ply")) in refuted_keys)]


def consolidate(store: RAGStore, threshold: float, dry_run: bool = False) -> dict:
    """整理经验库（离线，tidy_memory.py 调用）：
    1. 删除被证伪的 blunder 教训：同一个 blunder 的替代着法被 Stockfish 判为 mistake / blunder，
       说明模型给出的"应对原则"本身就是错的；
    2. 合并重复教训：同类教训两两相似度 ≥ threshold 的，只留最早的一条，seen 累加，
       其余的原文存进 meta.variants（见 rag.merge_into）。
    返回 {"refuted": [...], "merged": [(保留的, 被并入的), ...]}；dry_run 时只报告不改文件。"""
    entries = store.entries
    refuted = refuted_lessons(entries)
    gone = {id(e) for e in refuted}

    merged: list[tuple[dict, dict]] = []
    kept: list[tuple[dict, np.ndarray]] = []  # (保留的条目, 单位向量)
    for e in entries:
        if id(e) in gone or not is_lesson(e):
            continue
        v = np.array(e["emb"], dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-9
        target = next((k for k, kv in kept if kv.shape == v.shape and k["meta"].get("kind") == e["meta"].get("kind")
                       and float(kv @ v) >= threshold), None)
        if target is None:
            kept.append((e, v))
        else:
            merged.append((target, e))
            gone.add(id(e))
    report = {"refuted": refuted, "merged": merged}
    if dry_run:
        return report
    for keep, dup in merged:
        merge_into(keep, dup["text"], dup["meta"])
    if gone:
        store.remove(lambda e: id(e) in gone)
    else:
        store.save()
    return report


def recall_experience(board: chess.Board) -> list[dict]:
    """按"阶段 + 双方子力 + 最近着法"检索相关教训，直接放进 prompt。
    注意：这只是检索用的查询文本，不会展示给模型。"""
    if AUTO_RECALL_K <= 0:
        return []
    me = board.turn
    recent = " ".join(san_history(board).split()[-12:])
    q = (f"{game_phase(board)}，我执{COLOR_ZH[me]}，我方子力 {material_text(board, me)}，"
         f"对方子力 {material_text(board, not me)}{'，正被将军' if board.is_check() else ''}；最近着法 {recent}")
    try:
        return experience_rag.query(EMBED_QUERY_PREFIX + q, k=AUTO_RECALL_K, filter_fn=is_lesson)
    except Exception as e:
        print(f"[RECALL] failed: {e}")
        return []
