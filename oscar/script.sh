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

# Take the last line (prediction output) and fix single quotes
RAW_JSON=$(tail -n 1 "$RAW_LOG")
echo "$RAW_JSON" | sed "s/'/\"/g" > "$OUTPUT_JSON"

echo "✅ JSON saved to: $OUTPUT_JSON"