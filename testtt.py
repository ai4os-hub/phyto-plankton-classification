import requests
import json

url = "http://127.0.0.1:5000/api"

response = requests.get(url)

data = response.json()

print(json.dumps(data, indent=4))
