#!/bin/bash
# ── train.sh ─────────────────────────────────────────────────────────────────
# Full training pipeline: chuẩn bị dữ liệu + fine-tune Llama-3.1-8B QLoRA.
# Yêu cầu: GPU >= 14 GB VRAM (Tesla T4 / A10 / A100 ...)

set -e

DATA_DIR="${1:-sample_data}"
CONFIG="${2:-configs/train.yaml}"
OUTPUT_DIR="${3:-outputs}"
TRAIN_SAMPLES="${4:-3000}"

echo "===================================================="
echo "  Banking Intent Classification — Training Pipeline"
echo "===================================================="

# ── Step 1: Chuẩn bị dữ liệu ────────────────────────────────────────────────
echo ""
echo "=== [1/2] Chuẩn bị dữ liệu ==="
bash scripts/prepare_data.sh "$DATA_DIR" "$TRAIN_SAMPLES"

# ── Step 2: Fine-tune ─────────────────────────────────────────────────────────
echo ""
echo "=== [2/2] Fine-tune model ==="
python scripts/train.py \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "===================================================="
echo "  Training hoan tất!"
echo "  Model saved : $OUTPUT_DIR/banking_intent_model/"
echo "===================================================="