import base64

import requests
print("start")

endpoint = "https://inference.cloud.ai4eosc.eu/run/ai4papi-***********************"
token = "*************************************************************************"



def get_base64(fpath):
    with open(fpath, "rb") as f:
        encoded_str = base64.b64encode(f.read()).decode("utf-8")
    return encoded_str

data = {
    "oscar-files": [
        {
            "key": "image",
            "file_format": "jpg",
            "data": get_base64("oscar\plankton.jpg"),
        },
    ]
}
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}",
}
r = requests.post(url=endpoint, headers=headers, json=data)

if r.status_code == 401:
    raise Exception("Invalid token.")
if not r.ok:
    raise Exception(f"Some error has occurred: {r}")

print(r.text)
