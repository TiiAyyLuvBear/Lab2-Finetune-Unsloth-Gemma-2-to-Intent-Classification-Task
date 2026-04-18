"""
evaluate_model.py
=================
Đánh giá model đã fine-tune trên tập test:
  - Accuracy (độ chính xác)
  - Inference speed (tốc độ truy xuất câu trả lời)
  - Chi tiết từng prediction (opt-in)

Model được CACHE — không reload mỗi lần chạy evaluate.

Usage:
    # Đánh giá đầy đủ
    python scripts/evaluate_model.py -m outputs/banking_intent_model

    # Giới hạn 500 mẫu + hiển thị chi tiết
    python scripts/evaluate_model.py -m outputs/banking_intent_model -n 500 -d

    # Lưu kết quả ra JSON
    python scripts/evaluate_model.py -m outputs/banking_intent_model -o results.json
"""

import argparse
import json
import time
import yaml
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CACHE
# ═══════════════════════════════════════════════════════════════════════════════
_model_cache: dict[str, tuple] = {}  # model_path -> (model, tokenizer, device)


def get_cached_model_tokenizer(
    model_path: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    force_reload: bool = False,
):
    """Load model + tokenizer (cached)."""
    model_path = str(Path(model_path).resolve())
    cache_key = f"{model_path}|{max_seq_length}|{load_in_4bit}"

    if cache_key in _model_cache and not force_reload:
        print(f"[CACHE HIT] Model '{model_path}' đã được load — sử dụng cache.")
        return _model_cache[cache_key]

    print(f"[CACHE MISS] Loading model từ: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    device = model.device

    _model_cache[cache_key] = (model, tokenizer, device)
    print(f"[CACHE] Model loaded và cached tại key: {cache_key}")
    return model, tokenizer, device


def load_id2label(path: str) -> dict:
    """Load mapping label (số) → label_text."""
    df = pd.read_csv(path)
    return dict(zip(df["label"].astype(str), df["label_text"]))


def load_config(model_path: str, default_id2label: str) -> dict:
    """Load config từ thư mục model hoặc file inference.yaml."""
    config_path = Path(model_path) / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    alt_path = Path("configs/inference.yaml")
    if alt_path.exists():
        with open(alt_path, "r") as f:
            cfg = yaml.safe_load(f)
            cfg["data"] = {"id2label_path": default_id2label}
            return cfg

    return {
        "model": {"max_seq_length": 2048, "load_in_4bit": True},
        "generation": {"max_new_tokens": 64, "temperature": 0.0, "do_sample": False},
        "data": {"id2label_path": default_id2label},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ALPACA PROMPT (đồng bộ với train.py)
# ═══════════════════════════════════════════════════════════════════════════════
ALPACA_PROMPT = """Below is an utterance from a customer.
Your task is to classify this utterance into one of the 77 banking intents.

### Instruction:
Identify the intent of the following message.

### Input:
{}

### Response:
"""


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════
class Evaluator:
    """Đánh giá model: accuracy + inference speed."""

    def __init__(
        self,
        model_path: str,
        id2label_path: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
    ):
        if not UNSLOTH_AVAILABLE:
            raise RuntimeError("Unsloth is required. Cài: pip install unsloth")

        # ── Load model từ cache ──────────────────────────────────────────────
        self.model, self.tokenizer, self.device = get_cached_model_tokenizer(
            model_path, max_seq_length, load_in_4bit
        )
        self.id2label = load_id2label(id2label_path)
        self.pad_token_id = self.tokenizer.eos_token_id

    def _match_intent(self, generated: str) -> str:
        """Fuzzy match generated text với 77 intent labels."""
        generated = generated.strip()
        if generated in self.id2label.values():
            return generated
        for intent in self.id2label.values():
            if intent in generated or generated in intent:
                return intent
        return generated

    def predict(self, message: str) -> str:
        """Dự đoán intent cho một câu."""
        prompt = ALPACA_PROMPT.format(message)
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
            use_cache=True,
            pad_token_id=self.pad_token_id,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return self._match_intent(generated_text)

    @torch.no_grad()
    def evaluate(
        self,
        test_file: str,
        num_samples: Optional[int] = None,
        show_details: bool = False,
        output_file: Optional[str] = None,
    ) -> dict:
        """
        Đánh giá accuracy + speed trên tập test.

        Returns:
            dict với keys: accuracy, total, correct, avg_latency_ms, max_latency_ms, min_latency_ms
        """
        df_test = pd.read_csv(test_file)

        if num_samples and len(df_test) > num_samples:
            df_test = df_test.head(num_samples)

        total = len(df_test)
        correct = 0
        latencies = []  # ms per sample
        predictions = []

        print(f"Bắt đầu đánh giá trên {total} mẫu ...")

        for idx, row in df_test.iterrows():
            text = row["text"]
            true_label = self.id2label[str(row["label"])]

            # ── Measure inference time ────────────────────────────────────────
            start = time.perf_counter()
            pred_label = self.predict(text)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

            is_correct = pred_label == true_label
            if is_correct:
                correct += 1

            predictions.append({
                "idx": int(idx),
                "text": text[:80] + "..." if len(text) > 80 else text,
                "predicted": pred_label,
                "actual": true_label,
                "correct": is_correct,
                "latency_ms": round(elapsed_ms, 2),
            })

            if show_details:
                mark = "✓" if is_correct else "✗"
                print(f"[{mark}] {text[:60]:<60} | pred={pred_label:<25} | actual={true_label:<25}")

        # ── Summary stats ───────────────────────────────────────────────────────
        accuracy = correct / total * 100
        avg_lat = sum(latencies) / len(latencies)
        max_lat = max(latencies)
        min_lat = min(latencies)

        results = {
            "accuracy": round(accuracy, 2),
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "avg_latency_ms": round(avg_lat, 2),
            "max_latency_ms": round(max_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "predictions": predictions if show_details else predictions[:5],
        }

        # ── Print summary ───────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Accuracy        : {accuracy:.2f}%  ({correct}/{total})")
        print(f"  Avg Latency     : {avg_lat:.2f} ms/sample")
        print(f"  Max Latency     : {max_lat:.2f} ms")
        print(f"  Min Latency     : {min_lat:.2f} ms")
        print("=" * 60)

        # ── Save results ───────────────────────────────────────────────────────
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Kết quả đã lưu vào: {output_file}")

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá Banking Intent Model: Accuracy + Speed"
    )
    parser.add_argument("-m", "--model_path", type=str, required=True,
                        help="Thư mục chứa model checkpoint")
    parser.add_argument("-c", "--config", type=str, default="configs/inference.yaml",
                        help="File config inference YAML")
    parser.add_argument("-i", "--id2label_path", type=str, default="sample_data/id2label.csv",
                        help="File id2label.csv")
    parser.add_argument("-t", "--test_file", type=str, default="sample_data/test.csv",
                        help="File test CSV")
    parser.add_argument("-n", "--num_samples", type=int, default=None,
                        help="Giới hạn số mẫu test (mặc định: tất cả)")
    parser.add_argument("-d", "--show_details", action="store_true",
                        help="Hiển thị chi tiết từng prediction")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Lưu kết quả ra file JSON")
    parser.add_argument("--reload", action="store_true",
                        help="Force reload model (bỏ qua cache)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Banking Intent Model — Evaluation")
    print("=" * 60)

    # ── Load config ─────────────────────────────────────────────────────────────
    cfg = load_config(args.model_path, args.id2label_path)
    if Path(args.config).exists():
        with open(args.config, "r") as f:
            user_cfg = yaml.safe_load(f)
            cfg["model"].update(user_cfg.get("model", {}))
            cfg["generation"].update(user_cfg.get("generation", {}))

    # ── Load model (cache) ──────────────────────────────────────────────────────
    evaluator = Evaluator(
        model_path=args.model_path,
        id2label_path=args.id2label_path,
        max_seq_length=int(cfg["model"].get("max_seq_length", 2048)),
        load_in_4bit=bool(cfg["model"].get("load_in_4bit", True)),
    )
    print("Model loaded. Bắt đầu đánh giá ...")
    print("")

    # ── Run evaluation ─────────────────────────────────────────────────────────
    evaluator.evaluate(
        test_file=args.test_file,
        num_samples=args.num_samples,
        show_details=args.show_details,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()