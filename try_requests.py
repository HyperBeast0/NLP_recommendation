import requests

url = "http://127.0.0.1:8000/recommend"
payload = {"text": "Купить платье девушке на новый год"}
headers = {"Content-Type": "application/json"}

resp = requests.post(url, json=payload, headers=headers, timeout=20)

print("HTTP status:", resp.status_code)
print("Response headers:", resp.headers)
print("Response body:")
print(resp.text)

try:
    data = resp.json()
    print("Запрос: ", payload.values())
    print("Сущности:", data["entities"])
    print("Ключевые слова:", data["keywords"])
    print("Категории:", data["recommendations"]["categories"])

    for group in data["recommendations"]["products"]:
        print(f"\nКатегория: {group['category']}")
        for prod in group["products"]:
            print(f"- id: {prod['id']}, title: {prod['title']}")
except ValueError as e:
    print("Не смогли распарсить JSON:", e)
