import os

import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("LICHESS_TOKEN")
if not TOKEN:
    raise SystemExit("请先在 .env 中设置 LICHESS_TOKEN")

headers = {
    "Authorization": f"Bearer {TOKEN}"
}

r = requests.post(
    "https://lichess.org/api/bot/account/upgrade",
    headers=headers
)

print(r.status_code)
print(r.text)
