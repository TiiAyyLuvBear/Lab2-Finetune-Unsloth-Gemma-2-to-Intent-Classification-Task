#!/bin/bash
# ── inference.sh ─────────────────────────────────────────────────────────────
# Dự đoán intent cho một câu từ file.
# Gọi unified inference.py (chứa class IntentClassification + CLI)

set -e

# ── Defaults ─────────────────────────────────────────────────────────────────
MODEL_PATH="outputs/banking_intent_model"
QUERY=""
INPUT_FILE=""
OUTPUT_FILE=""

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model-path   Path to fine-tuned model directory (default: outputs/banking_intent_model)"
    echo "  -q, --query        Single query text to classify"
    echo "  -f, --input-file   File .txt chứa nhiều câu (mỗi dòng = 1 câu)"
    echo "  -o, --output-file  Save results to JSON file"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -m outputs/banking_intent_model"
    echo "  $0 -m outputs/banking_intent_model -q \"I want to transfer money\""
    echo "  $0 -m outputs/banking_intent_model -f queries.txt -o results.json"
    echo ""
    exit 0
}

# ── Parse named arguments ──────────────────────────────────────────────────────
while getopts "m:q:f:o:h" opt; do
    case $opt in
        m)  MODEL_PATH="$OPTARG" ;;
        q)  QUERY="$OPTARG" ;;
        f)  INPUT_FILE="$OPTARG" ;;
        o)  OUTPUT_FILE="$OPTARG" ;;
        h)  usage ;;
        \?) echo "Unknown option: -$OPTARG" >&2; usage ;;
    esac
done

echo "===================================================="
echo "  Banking Intent Classification — Inference"
echo "===================================================="
echo "  model_path  : $MODEL_PATH"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python scripts/inference.py -m \"$MODEL_PATH\""

if [ -n "$QUERY" ]; then
    CMD="$CMD -q \"$QUERY\""
fi

if [ -n "$INPUT_FILE" ]; then
    CMD="$CMD -f \"$INPUT_FILE\""
fi

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD -o \"$OUTPUT_FILE\""
fi

# ── Run inference ─────────────────────────────────────────────────────────────
eval $CMD