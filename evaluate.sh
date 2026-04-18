#!/bin/bash
# ── evaluate.sh ───────────────────────────────────────────────────────────────
# Đánh giá model: accuracy + inference speed.
# Yêu cầu: model đã được train và lưu tại outputs/banking_intent_model/

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_PATH="outputs/banking_intent_model"
CONFIG="configs/inference.yaml"
ID2LABEL_PATH="sample_data/id2label.csv"
TEST_FILE="sample_data/test.csv"
NUM_SAMPLES="1000"
SHOW_DETAILS=false
OUTPUT=""

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model-path      Path to model checkpoint directory   (default: outputs/banking_intent_model)"
    echo "  -c, --config          Path to inference config YAML        (default: configs/inference.yaml)"
    echo "  -i, --id2label-path   Path to id2label.csv                  (default: sample_data/id2label.csv)"
    echo "  -t, --test-file       Path to test CSV file                 (default: sample_data/test.csv)"
    echo "  -n, --num-samples     Number of test samples to evaluate    (default: 1000)"
    echo "  -d, --show-details    Show per-sample prediction details"
    echo "  -o, --output          Save results to JSON file             (default: none)"
    echo "  -h, --help            Show this help message"
    echo ""
    exit 0
}

# ── Parse named arguments ──────────────────────────────────────────────────────
while getopts "m:c:i:t:n:do:h" opt; do
    case $opt in
        m)  MODEL_PATH="$OPTARG" ;;
        c)  CONFIG="$OPTARG" ;;
        i)  ID2LABEL_PATH="$OPTARG" ;;
        t)  TEST_FILE="$OPTARG" ;;
        n)  NUM_SAMPLES="$OPTARG" ;;
        d)  SHOW_DETAILS=true ;;
        o)  OUTPUT="$OPTARG" ;;
        h)  usage ;;
        \?) echo "Unknown option: -$OPTARG" >&2; usage ;;
    esac
done

echo "===================================================="
echo "  Banking Intent Model — Evaluation"
echo "===================================================="
echo "  model_path      : $MODEL_PATH"
echo "  config          : $CONFIG"
echo "  id2label_path   : $ID2LABEL_PATH"
echo "  test_file       : $TEST_FILE"
echo "  num_samples     : ${NUM_SAMPLES:-all}"
echo "  show_details    : $SHOW_DETAILS"
echo "  output          : ${OUTPUT:-none}"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python3 scripts/evaluate_model.py --model_path $MODEL_PATH --id2label_path $ID2LABEL_PATH --test_file $TEST_FILE"

if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

if [ "$SHOW_DETAILS" = true ]; then
    CMD="$CMD --show_details"
fi

if [ -n "$OUTPUT" ]; then
    CMD="$CMD --output $OUTPUT"
fi

# ── Run evaluation ─────────────────────────────────────────────────────────────
echo "=== Bắt đầu đánh giá ==="
eval $CMD