#!/bin/bash
# ── train.sh ─────────────────────────────────────────────────────────────────
# Full training pipeline: chuẩn bị dữ liệu + fine-tune Gemma2-9B QLoRA.
# Yêu cầu: GPU >= 14 GB VRAM (Tesla T4 / A10 / A100 ...)

set -e

# ── Defaults ─────────────────────────────────────────────────────────────────
DATA_DIR="sample_data"
CONFIG="configs/train.yaml"
OUTPUT_DIR="outputs"
TRAIN_SAMPLES=3000
RESUME=false

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --data-dir      Directory containing input data      (default: sample_data)"
    echo "  -c, --config        Path to training config YAML file    (default: configs/train.yaml)"
    echo "  -o, --output-dir    Output directory for model artifacts (default: outputs)"
    echo "  -s, --train-samples Number of training samples to use    (default: 3000)"
    echo "  -r, --resume        Skip data prep, continue training if model exists"
    echo "  -h, --help          Show this help message"
    echo ""
    exit 0
}

# ── Parse named arguments ──────────────────────────────────────────────────────
while getopts "d:c:o:s:rh" opt; do
    case $opt in
        d)  DATA_DIR="$OPTARG" ;;
        c)  CONFIG="$OPTARG" ;;
        o)  OUTPUT_DIR="$OPTARG" ;;
        s)  TRAIN_SAMPLES="$OPTARG" ;;
        r)  RESUME=true ;;
        h)  usage ;;
        \?) echo "Unknown option: -$OPTARG" >&2; usage ;;
    esac
done

echo "===================================================="
echo "  Banking Intent Classification — Training Pipeline"
echo "===================================================="
echo "  data_dir       : $DATA_DIR"
echo "  config         : $CONFIG"
echo "  output_dir     : $OUTPUT_DIR"
echo "  train_samples  : $TRAIN_SAMPLES"
echo "  resume         : $RESUME"
echo ""

# ── Step 1: Chuẩn bị dữ liệu (bỏ qua nếu --resume) ────────────────────────────
if [ "$RESUME" = false ]; then
    echo "=== [1/2] Chuẩn bị dữ liệu ==="
    python scripts/preprocess_data.py \
        --data_dir "$DATA_DIR" \
        --train_samples "$TRAIN_SAMPLES"
else
    echo "=== [1/2] Chuẩn bị dữ liệu === [SKIPPED — resume mode]"
fi

# ── Step 2: Fine-tune ──────────────────────────────────────────────────────────
echo ""
echo "=== [2/2] Fine-tune model ==="
python scripts/train.py \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --resume

echo ""
echo "===================================================="
echo "  Training hoàn tất!"
echo "  Model saved : $OUTPUT_DIR/banking_intent_model/"
echo "===================================================="