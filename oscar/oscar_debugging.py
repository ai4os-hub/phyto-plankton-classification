import base64
import io
import json
import re
import zipfile
import ast
from pathlib import Path

import requests

print("start")

# token = "PUT_YOUR_TOKEN_HERE"
endpoint = "https://inference-walton.cloud.imagine-ai.eu/run/phyto-plankton-classification"
token = "************************************************"

image_path = Path(r"plankton.jpg")
out_dir = Path("oscar_response")
out_dir.mkdir(parents=True, exist_ok=True)

def get_base64_file(fpath: Path) -> str:
    with open(fpath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

data = {
    "oscar-files": [
        {"key": "image", "file_format": "jpg", "data": get_base64_file(image_path)},
    ]
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}",
}

r = requests.post(endpoint, headers=headers, json=data, timeout=300)

print("HTTP status:", r.status_code)
print("Content-Type:", r.headers.get("content-type"))

if r.status_code == 401:
    raise Exception("Invalid token (401).")
if not r.ok:
    raise Exception(f"Request failed: {r.status_code}\n{r.text}")

resp = r.text

# Find base64 ZIP blob even if logs come before it
m = re.search(r"(UEsDB[0-9A-Za-z+/=\r\n]+)", resp)
if not m:
    print("Could not find a base64 ZIP blob in the response. Raw response:")
    print(resp)
    raise SystemExit(1)

b64_zip = "".join(m.group(1).splitlines())
zip_bytes = base64.b64decode(b64_zip, validate=False)

with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
    names = z.namelist()
    print("Files returned:", names)
    z.extractall(out_dir)

print(f"Extracted to: {out_dir.resolve()}")

def parse_result_file(path: Path):
    """
    Try strict JSON first. If that fails, try parsing python-dict-like output.
    If everything fails, print the raw content for debugging.
    """
    raw = path.read_text(encoding="utf-8", errors="replace").strip()

    # 1) strict JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"\n⚠️ JSON parse failed for {path.name}: {e}")
        print("\n--- Raw file content (first 400 chars) ---")
        print(raw[:400])
        print("--- end preview ---\n")

    # 2) common OSCAR/deepaas pattern: python dict string using single quotes
    # Example: {'status': 'error', 'message': '...'}
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass

    # 3) sometimes logs + json are mixed; try to extract a {...} block
    m2 = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
    if m2:
        candidate = m2.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                obj = ast.literal_eval(candidate)
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                pass

    return {"status": "error", "message": "Could not parse output file", "raw": raw[:2000]}

# Read and print result
json_files = sorted(out_dir.glob("output_*.json"))
if json_files:
    jf = json_files[-1]
    result = parse_result_file(jf)
    print("\n=== Parsed service result ===\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
else:
    print("\nNo output_*.json found.\n")

# Print tail of raw log (best place to find the real service traceback)
txt_files = sorted(out_dir.glob("output_raw_*.txt"))
if txt_files:
    tf = txt_files[-1]
    print("\n=== Tail of raw log:", tf.name, "===\n")
    lines = tf.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(lines[-120:]))
else:
    print("\nNo output_raw_*.txt found.\n")