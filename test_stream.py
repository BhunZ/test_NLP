import requests
import json

url = "http://127.0.0.1:8000/api/v1/ask/stream"
payload = {
    "query": "Cơ chế attention là gì?",
    "top_k": 3,
    "llm_provider": "groq",
    "rerank": False,
    "enable_rewrite": False
}

print(f"Calling {url}...")
try:
    response = requests.post(url, json=payload, stream=True)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'))
    else:
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
