"""Lichess 主循环：接受挑战、跟随对局流、轮到我方时调 LLM 落子，终局后触发复盘。"""
import os
import queue
import threading
import time
from datetime import datetime

import berserk
import chess

from .boardtext import build_pgn, san_history
from .config import LICHESS_TOKEN, WAIT_TIMEOUT_SEC
from .live import live
from .player import safe_llm_move
from .review import record_snapshot, run_post_game


def run_lichess():
    client = berserk.Client(session=berserk.TokenSession(LICHESS_TOKEN))
    print(f"Waiting for games ... (idle timeout: {WAIT_TIMEOUT_SEC}s)")
    my_username = client.account.get()["username"].lower()
    print(f"My username: {my_username}")


    def _event_producer(q: "queue.Queue"):
        try:
            for ev in client.bots.stream_incoming_events():
                q.put(ev)
        except Exception as e:
            q.put({"__error__": str(e)})


    MAX_STREAM_RETRIES = 6


    def resilient_game_stream(game_id: str):
        """包装 stream_game_state：断线 / 流意外关闭时指数退避重连，
        重连后 Lichess 会重发 gameFull，主循环据此恢复局面。收到终局状态才结束。"""
        retries = 0
        while True:
            try:
                for state in client.bots.stream_game_state(game_id):
                    retries = 0
                    yield state
                    status = state.get("status") or (state.get("state") or {}).get("status", "started")
                    if status not in ("started", "created"):
                        return
                print(f"[STREAM] game {game_id} stream closed before finish, reconnecting")
            except Exception as e:
                print(f"[STREAM] game {game_id} stream error: {e}")
            retries += 1
            if retries > MAX_STREAM_RETRIES:
                raise RuntimeError(f"game stream lost after {MAX_STREAM_RETRIES} reconnects")
            wait = min(2 ** retries, 20)
            print(f"[STREAM] reconnect in {wait}s ({retries}/{MAX_STREAM_RETRIES})")
            time.sleep(wait)


    def handle_challenge(ev: dict):
        """自动接受标准规则的挑战，其余拒绝。"""
        ch = ev.get("challenge", {})
        cid = ch.get("id")
        challenger = (ch.get("challenger") or {}).get("name", "?")
        variant = (ch.get("variant") or {}).get("key", "standard")
        try:
            if variant != "standard":
                client.bots.decline_challenge(cid, reason="standard")
                print(f"[CHALLENGE] declined {cid} from {challenger} (variant={variant})")
            else:
                client.bots.accept_challenge(cid)
                print(f"[CHALLENGE] accepted {cid} from {challenger}")
        except Exception as e:
            print(f"[CHALLENGE] handle {cid} failed: {e}")


    event_queue: "queue.Queue" = queue.Queue()
    threading.Thread(target=_event_producer, args=(event_queue,), daemon=True).start()

    while True:
        try:
            event = event_queue.get(timeout=WAIT_TIMEOUT_SEC)
        except queue.Empty:
            print(f"No game in {WAIT_TIMEOUT_SEC}s, exit.")
            break

        if "__error__" in event:
            print(f"[ERROR] event stream: {event['__error__']}")
            break

        if event.get("type") == "challenge":
            handle_challenge(event)
            continue

        if event.get("type") != "gameStart":
            continue

        game_id = event["game"]["id"]
        print(f"=== Game started: {game_id} ===")

        is_white = None
        move_log = []
        snapshots: list = []  # 本局待验证的局面快照
        last_moves_str = None
        chat_messages: list = []  # [{"username","text","room","time","fen","ply","last_move"}]
        current_fen = chess.STARTING_FEN
        current_history = ""  # 当前局面之前的 SAN 走法，给聊天复盘用
        current_ply = 0
        current_last_move = None

        try:
            for state in resilient_game_stream(game_id):
                stype = state.get("type")

                if stype == "gameFull":
                    last_moves_str = None  # 重连后会重发 gameFull，允许重新决策当前手
                    white_name = state["white"].get("name", "").lower()
                    is_white = (white_name == my_username)
                    print(f"I play {'WHITE' if is_white else 'BLACK'}")
                    opp = state["black" if is_white else "white"]
                    live.start_game(game_id, "lichess", "白" if is_white else "黑",
                                    opp.get("name") or opp.get("id") or "对手")
                    moves_str = state["state"]["moves"]
                    game_status = state["state"].get("status", "started")
                    # gameFull 里也可能带已有 chatLines
                    for c in state.get("chatLines", []) or []:
                        chat_messages.append({
                            "username": c.get("username", ""),
                            "text": c.get("text", ""),
                            "room": c.get("room", "player"),
                            "time": datetime.now().isoformat(),
                            "fen": current_fen,
                            "history": current_history,
                            "ply": current_ply,
                            "last_move": current_last_move,
                        })
                        print(f"[CHAT<<] [{c.get('room','player')}] "
                              f"{c.get('username','')}: {c.get('text','')}")
                elif stype == "gameState":
                    moves_str = state["moves"]
                    game_status = state.get("status", "started")
                elif stype == "chatLine":
                    # Lichess 原生推送，无需轮询
                    uname = state.get("username", "")
                    text = state.get("text", "")
                    room = state.get("room", "player")
                    chat_messages.append({
                        "username": uname, "text": text, "room": room,
                        "time": datetime.now().isoformat(),
                        "fen": current_fen,
                            "history": current_history,
                        "ply": current_ply,
                        "last_move": current_last_move,
                    })
                    print(f"[CHAT<<] [{room}] {uname}: {text} "
                          f"(ply={current_ply}, last={current_last_move})")
                    continue
                else:
                    continue

                board = chess.Board()
                uci_list = moves_str.split() if moves_str else []
                for uci in uci_list:
                    try:
                        board.push_uci(uci)
                    except Exception:
                        pass

                # 更新"当前局面"快照，供之后的 chatLine 事件关联上下文
                current_fen = board.fen()
                current_history = san_history(board)
                live.sync(uci_list)
                current_ply = board.ply()
                current_last_move = uci_list[-1] if uci_list else None

                # 对局结束
                if game_status not in ("started", "created"):
                    print(f"Game finished, status={game_status}")
                    winner = state.get("winner")
                    if winner == "white":
                        result = "1-0"
                    elif winner == "black":
                        result = "0-1"
                    else:
                        result = "1/2-1/2"
                    live.set_status("reviewing", result)
                    pgn_text = build_pgn(uci_list, result)
                    os.makedirs("games", exist_ok=True)
                    pgn_path = os.path.join(
                        "games",
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{game_id}.pgn"
                    )
                    with open(pgn_path, "w", encoding="utf-8") as f:
                        f.write(pgn_text)
                    print(f"PGN saved to {pgn_path}")

                    my_color = "白" if is_white else "黑"
                    run_post_game(pgn_text, result, my_color, move_log, snapshots, uci_list, is_white,
                                  chat_messages=chat_messages, my_username=my_username)
                    live.set_status("finished", result)
                    break

                if is_white is None:
                    continue

                my_turn = (board.turn == chess.WHITE and is_white) or \
                          (board.turn == chess.BLACK and not is_white)
                if not my_turn:
                    continue

                if moves_str == last_moves_str:
                    continue
                last_moves_str = moves_str

                print(f"\n--- ply {board.ply()+1} my move ---")
                print(board)

                # 构造「对方走子前」局面
                prev_board = None
                opp_last_move = None
                if uci_list:
                    prev_board = chess.Board()
                    for uci in uci_list[:-1]:
                        try:
                            prev_board.push_uci(uci)
                        except Exception:
                            pass
                    opp_last_move = uci_list[-1]

                # LLM 接口异常也不能让这盘棋超时判负：safe_llm_move 出错时随机走合法着法
                move, think, opp_intent, obs = safe_llm_move(
                    board, board.ply() + 1, prev_board, opp_last_move,
                    chat_messages=chat_messages,
                )
                if opp_intent:
                    print(f"[OPP]   {opp_intent}")
                if obs:
                    for k, v in obs.items():
                        if v:
                            print(f"[OBS]   {k}: {str(v)[:200]}")
                print(f"[THINK] {think}")
                print(f"[MOVE]  {move}")

                try:
                    client.bots.make_move(game_id, move)
                    move_log.append((board.ply() + 1, board.fen(), move, think))
                    print(f"Played: {move}")
                except Exception as e:
                    print(f"Move failed: {e}")

                # 局面快照只先缓存在内存，赛后经 Stockfish 验证再决定是否写入经验库
                record_snapshot(snapshots, board, move, obs)

                time.sleep(0.5)

        except Exception as e:
            print(f"[ERROR] game loop: {e}")
            continue

        print(f"Waiting for next game ... (idle timeout: {WAIT_TIMEOUT_SEC}s)")
