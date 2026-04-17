"""
inference.py
Chạy inference với model banking-intent đã fine-tune.
Hỗ trợ:
  - Dự đoán đơn lẻ (--input_text)
  - Đọc file .txt/.csv chứa nhiều câu (--input_file)
  - Tính accuracy trên test.csv (--evaluate)
  - Hiển thị mẫu dự đoán kèm nhãn thực tế (--show_samples)
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Prompt template (đồng bộ với train.py) ────────────────────────────────────
ALPACA_PROMPT = """Below is an utterance from a customer.
Your task is to classify this utterance into one of the 77 banking intents.

### Instruction:
Identify the intent of the following message.

### Input:
{}

### Response:
"""


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_id2label(path: str) -> dict:
    """Load mapping label (số) → label_text từ CSV."""
    df = pd.read_csv(path)
    return dict(zip(df["label"].astype(str), df["label_text"]))


class IntentClassifier:
    """Wrapper dự đoán intent cho banking domain."""

    def __init__(self, model_path: str, id2label_path: str = "sample_data/id2label.csv"):
        if not UNSLOTH_AVAILABLE:
            raise RuntimeError("Unsloth is required. Cài: pip install unsloth")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

        self.id2label = load_id2label(id2label_path)  # key = "0" → "lost_or_stolen_card"
        self.prompt_style = ALPACA_PROMPT

    def __call__(self, text: str) -> str:
        """Dự đoán intent cho một câu duy nhất. Trả về intent TEXT."""
        inputs = self.tokenizer(
            [self.prompt_style.format(text)],
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            use_cache=True,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Cắt bỏ phần prompt, chỉ giữ phần generated
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        intent_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # ── Fuzzy match: so sánh với 77 intent labels ───────────────────────
        if intent_text in self.id2label.values():
            return intent_text

        for intent_name in self.id2label.values():
            if intent_name in intent_text or intent_text in intent_name:
                return intent_name

        return intent_text  # fallback: trả về text thuần túy


@torch.no_grad()
def evaluate(
    model_path: str,
    id2label_path: str = "sample_data/id2label.csv",
    test_file: str = "sample_data/test.csv",
    num_samples: int = None,
) -> float:
    """Tính accuracy trên tập test. Trả về accuracy (%)."""
    classifier = IntentClassifier(model_path, id2label_path)
    df_test = pd.read_csv(test_file)
    id2label = load_id2label(id2label_path)

    if num_samples and len(df_test) > num_samples:
        df_test = df_test.head(num_samples)

    correct = 0
    total = len(df_test)
    logger.info("Đánh giá trên %d mẫu ...", total)

    for _, row in df_test.iterrows():
        pred = classifier.predict(row["text"])
        true = id2label[str(row["label"])]
        if pred == true:
            correct += 1

    accuracy = correct / total * 100
    logger.info("Accuracy: %.2f%%  (%d/%d)", accuracy, correct, total)
    return accuracy


@torch.no_grad()
def show_samples(
    model_path: str,
    id2label_path: str = "sample_data/id2label.csv",
    test_file: str = "sample_data/test.csv",
    num_samples: int = 10,
    seed: int = 42,
) -> None:
    """Hiển thị num_samples dự đoán kèm nhãn thực tế (dùng random seed)."""
    classifier = IntentClassifier(model_path, id2label_path)
    df_test = pd.read_csv(test_file).sample(n=num_samples, random_state=seed)
    id2label = load_id2label(id2label_path)

    header = f"{'Text':<48} | {'Predicted':<32} | {'Actual':<32} | ✓/✗"
    print(header)
    print("-" * len(header))

    for _, row in df_test.iterrows():
        pred = classifier.predict(row["text"])
        true = id2label[str(row["label"])]
        short_text = row["text"][:45] + "..." if len(row["text"]) > 45 else row["text"]
        match = "✓" if pred == true else "✗"
        print(f"{short_text:<48} | {pred:<32} | {true:<32} | {match}")


def main():
    parser = argparse.ArgumentParser(description="Inference với Banking Intent Model.")
    parser.add_argument("--config", type=str, default="configs/inference.yaml",
                        help="Đường dẫn file inference config YAML")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Đường dẫn thư mục model (adapter)")
    parser.add_argument("--id2label_path", type=str, default="sample_data/id2label.csv",
                        help="File chứa id → label_text mapping")
    parser.add_argument("--input_text", type=str, default=None,
                        help="Một câu duy nhất cần phân loại")
    parser.add_argument("--input_file", type=str, default=None,
                        help="File .txt chứa nhiều câu (mỗi dòng = 1 câu)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Lưu kết quả ra file JSON")
    parser.add_argument("--evaluate", action="store_true",
                        help="Tính accuracy trên tập test")
    parser.add_argument("--test_file", type=str, default="sample_data/test.csv",
                        help="File test CSV (dùng với --evaluate)")
    parser.add_argument("--num_eval_samples", type=int, default=None,
                        help="Giới hạn số mẫu test khi đánh giá")
    parser.add_argument("--show_samples", action="store_true",
                        help="Hiển thị một số mẫu dự đoán kèm nhãn thực tế")
    parser.add_argument("--num_show_samples", type=int, default=10,
                        help="Số mẫu hiển thị (với --show_samples)")
    args = parser.parse_args()

    # Load config (không bắt buộc nếu dùng CLI args)
    if Path(args.config).exists():
        cfg = load_config(args.config)
    else:
        cfg = {}

    # ── Batch / single prediction ─────────────────────────────────────────────
    queries = []
    if args.input_text:
        queries = [args.input_text]
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
        logger.info("Đã đọc %d câu từ %s", len(queries), args.input_file)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    if args.evaluate:
        evaluate(
            model_path=args.model_path,
            id2label_path=args.id2label_path,
            test_file=args.test_file,
            num_samples=args.num_eval_samples,
        )
        return  # evaluate đã in kết quả, không cần làm gì thêm

    # ── Show samples ───────────────────────────────────────────────────────────
    if args.show_samples:
        show_samples(
            model_path=args.model_path,
            id2label_path=args.id2label_path,
            test_file=args.test_file,
            num_samples=args.num_show_samples,
        )
        return

    # ── Interactive / batch inference ─────────────────────────────────────────
    if not queries:
        print("Chế độ tương tác — nhập câu hỏi banking (Ctrl+C để thoát):")
        while True:
            try:
                q = input("Query > ").strip()
                if q:
                    queries.append(q)
            except (EOFError, KeyboardInterrupt):
                break

    classifier = IntentClassifier(args.model_path, args.id2label_path)
    results = []
    for q in queries:
        intent = classifier.predict(q)
        results.append({"query": q, "predicted_intent": intent})
        print(f"Query  : {q}")
        print(f"Intent : {intent}")
        print("---")

    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Đã lưu kết quả vào %s", args.output_file)


if __name__ == "__main__":
    main()