import requests

TOKEN = "你的token"

headers = {
    "Authorization": f"Bearer {TOKEN}"
}

r = requests.post(
    "https://lichess.org/api/bot/account/upgrade",
    headers=headers
)

print(r.status_code)
print(r.text)