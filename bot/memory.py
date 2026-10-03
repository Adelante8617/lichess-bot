"""RAG：开局库 + 经验记忆库，以及每步的经验自动召回。"""
import chess
from openai import OpenAI

from .boardtext import COLOR_ZH, game_phase, material_text, san_history
from .config import (AUTO_RECALL_K, EMBED_API_KEY, EMBED_BACKEND, EMBED_BASE_URL, EMBED_DEVICE,
                     EMBED_LOCAL_MODEL, EMBED_MODEL, EMBED_QUERY_PREFIX)
from .rag import RAGStore

_emb_client = None
_local_embedder = None


def embed(text: str):
    global _emb_client, _local_embedder
    if EMBED_BACKEND == "local":
        if _local_embedder is None:
            from sentence_transformers import SentenceTransformer
            print(f"[EMBED] loading local model {EMBED_LOCAL_MODEL} (device={EMBED_DEVICE or 'auto'}) ...")
            _local_embedder = SentenceTransformer(EMBED_LOCAL_MODEL, device=EMBED_DEVICE)
        return _local_embedder.encode(text, normalize_embeddings=True).tolist()
    if _emb_client is None:
        _emb_client = OpenAI(api_key=EMBED_API_KEY, base_url=EMBED_BASE_URL)
    resp = _emb_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


opening_rag = RAGStore("data/openings.jsonl", embed)
experience_rag = RAGStore("data/experience.jsonl", embed)


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
