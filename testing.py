import requests
r = requests.get(
    "https://en.wikipedia.org/w/api.php",
    params={"action": "query", "list": "search", "srsearch": "Nikola Tesla",
            "format": "json", "srlimit": 3},
    headers={"User-Agent": "DocRagApp/1.0 (tanyaraj381@gmail.com)"},
    timeout=15,
)
print(r.status_code, r.text[:200])