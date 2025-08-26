#!/bin/sh
set -eu

: "${INPUT_FILE_PATH:?Need INPUT_FILE_PATH}"

TMP_OUTPUT_DIR="${TMP_OUTPUT_DIR:-/tmp}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

RAW_LOG="$TMP_OUTPUT_DIR/output_raw_$TIMESTAMP.txt"
OUTPUT_JSON="$TMP_OUTPUT_DIR/output_$TIMESTAMP.json"

deepaas-cli predict \
  --image "$INPUT_FILE_PATH" \
  --timestamp Phytoplankton_EfficientNetV2B0 \
  --ckpt_name final_model.h5 2>&1 | tee "$RAW_LOG"

LAST_RETURN=$(grep "return:" "$RAW_LOG" | tail -n 1 | sed -E "s/^.*return: //")

python3 -c "
import json, re, ast
raw = '''$LAST_RETURN'''
raw_clean = re.sub(r\"np\.str_\('([^']+)'\)\", r'\"\\1\"', raw)
data = ast.literal_eval(raw_clean)
print(json.dumps(data, indent=2))
" > "$OUTPUT_JSON"

echo "✅ JSON saved to: $OUTPUT_JSON"
