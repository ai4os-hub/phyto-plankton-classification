#!/bin/sh
set -eu

: "${INPUT_FILE_PATH:?Need INPUT_FILE_PATH}"

TMP_OUTPUT_DIR="${TMP_OUTPUT_DIR:-/tmp}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

RAW_LOG="$TMP_OUTPUT_DIR/output_raw_$TIMESTAMP.txt"
OUTPUT_JSON="$TMP_OUTPUT_DIR/output_$TIMESTAMP.json"

# Run prediction and store raw log
deepaas-cli predict \
  --image "$INPUT_FILE_PATH" \
  --timestamp Phytoplankton_EfficientNetV2B0 \
  --ckpt_name final_model.h5 2>&1 | tee "$RAW_LOG"

# Extract the line containing the result (more reliable than "return:")
LAST_RETURN=$(grep "{'status':" "$RAW_LOG" | tail -n 1 || true)

# Remove ANSI escape codes (colors etc.)
LAST_RETURN_CLEAN=$(echo "${LAST_RETURN:-}" | sed -r 's/\x1B\[[0-9;]*[mK]//g')

# Debug (optional, can remove later)
echo "------ PARSED RETURN ------"
echo "$LAST_RETURN_CLEAN"
echo "---------------------------"

# Convert to JSON
python3 -c "
import json, re, ast

raw = '''$LAST_RETURN_CLEAN'''

# Remove numpy string wrappers like np.str_('abc')
raw = re.sub(r\"np\.str_\('([^']+)'\)\", r'\"\\1\"', raw)

try:
    data = ast.literal_eval(raw)
except Exception as e:
    print('FAILED TO PARSE:')
    print(raw)
    raise

print(json.dumps(data, indent=2))
" > "$OUTPUT_JSON"

echo "✅ JSON saved to: $OUTPUT_JSON"
