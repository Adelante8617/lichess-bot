"""
对局观战页面：读取游戏进程写出的 live/state.json，在浏览器里展示。
只依赖标准库，与游戏进程互相独立——先开它、后开它都行，本地对局和 Lichess 对局通用。

用法：
  python viewer.py                 # 浏览器打开 http://127.0.0.1:8000
  python viewer.py --port 9000 --no-browser
"""
import argparse
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from live import LIVE_PATH

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "web", "index.html")
EMPTY = json.dumps({"status": "idle", "updated_at": 0}).encode()


class Handler(BaseHTTPRequestHandler):
    state_path = LIVE_PATH

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            with open(INDEX, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/state":
            try:
                with open(self.state_path, "rb") as f:
                    body = f.read()
                json.loads(body)  # 写入瞬间读到半截文件时，退回空状态而不是报错
            except (OSError, ValueError):
                body = EMPTY
            self._send(200, body, "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *args):  # 不刷屏
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--state", default=LIVE_PATH, help="状态文件路径")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    Handler.state_path = args.state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"观战页面: {url}   (状态文件: {args.state})   Ctrl+C 退出")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
