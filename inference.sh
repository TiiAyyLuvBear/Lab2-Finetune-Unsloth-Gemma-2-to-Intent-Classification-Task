#!/bin/bash
# ── inference.sh ─────────────────────────────────────────────────────────────
# Dự đoán intent cho một câu hoặc nhiều câu từ file.

set -e

MODEL_PATH="${1:-banking_intent_model}"
CONFIG="${2:-configs/inference.yaml}"
QUERY="${3:-}"

echo "=== Banking Intent Inference ==="
echo "  model_path: $MODEL_PATH"
echo "  config    : $CONFIG"

if [ -n "$QUERY" ]; then
    # ── Single query mode ─────────────────────────────────────────────────────
    echo ""
    echo "Input : $QUERY"
    python scripts/inference.py \
        --config "$CONFIG" \
        --model_path "$MODEL_PATH" \
        --input_text "$QUERY"
else
    # ── Interactive mode ──────────────────────────────────────────────────────
    echo ""
    echo "Khong co query truyen vao. Chuyen sang che do tuong tac."
    echo "Nhập câu hỏi banking (Ctrl+C để thoát):"
    python scripts/inference.py \
        --config "$CONFIG" \
        --model_path "$MODEL_PATH"
fi