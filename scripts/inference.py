"""
inference.py
=====================
Unified inference file cho Banking Intent Classification.
Model được CACHE sau lần load đầu tiên — không reload mỗi lần gọi.

Class API:
    from scripts.intent_classifier import IntentClassification
    classifier = IntentClassification("outputs/banking_intent_model")
    label = classifier("I want to transfer money")

CLI:
    python3 scripts/inference.py -m outputs/banking_intent_model
    python3 scripts/inference.py -m outputs/banking_intent_model -q "I lost my card"
"""

import argparse
import json
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
# MODEL CACHE — singleton, không reload model mỗi lần gọi
# ═══════════════════════════════════════════════════════════════════════════════
_model_cache: dict[str, "IntentClassification"] = {}
_tokenizer_cache: dict = {}  # backup if needed


def get_cached_model(model_path: str, force_reload: bool = False) -> "IntentClassification":
    """Load model (cached) hoặc trả về instance đã cache."""
    model_path = str(Path(model_path).resolve())

    if model_path in _model_cache and not force_reload:
        print(f"[CACHE HIT] Model '{model_path}' đã được load — sử dụng cache.")
        return _model_cache[model_path]

    print(f"[CACHE MISS] Loading model từ: {model_path}")
    classifier = IntentClassification(model_path)
    _model_cache[model_path] = classifier
    return classifier


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
# INTENT CLASSIFICATION CLASS
# ═══════════════════════════════════════════════════════════════════════════════
class IntentClassification:
    """
    Inference class cho Banking Intent Classification.

    Args:
        model_path: Thư mục chứa model checkpoint (adapter).
    """

    def __init__(self, model_path: str):
        if not UNSLOTH_AVAILABLE:
            raise RuntimeError("Unsloth is required. Cài đặt: pip install unsloth")

        self.model_path = Path(model_path).resolve()

        # ── Load config ────────────────────────────────────────────────────────
        config_path = self.model_path / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = {
                "model": {"max_seq_length": 2048, "load_in_4bit": True},
                "generation": {
                    "max_new_tokens": 64,
                    "temperature": 0.0,
                    "do_sample": False,
                    "top_p": 1.0,
                    "use_cache": True,
                },
                "data": {
                    "id2label_path": str(self.model_path.parent.parent / "sample_data" / "id2label.csv"),
                }
            }

        # ── Load id2label mapping ──────────────────────────────────────────────
        id2label_path = self.cfg.get("data", {}).get("id2label_path", "sample_data/id2label.csv")
        if not Path(id2label_path).exists():
            id2label_path = self.model_path.parent.parent / id2label_path
        if not Path(id2label_path).exists():
            raise FileNotFoundError(f"Không tìm thấy id2label.csv tại: {id2label_path}")

        df = pd.read_csv(id2label_path)
        self.id2label = dict(zip(df["label"].astype(str), df["label_text"]))
        print(f"[IntentClassification] Loaded {len(self.id2label)} intent labels from {id2label_path}")

        # ── Load model + tokenizer ────────────────────────────────────────────
        m_cfg = self.cfg["model"]
        print(f"[IntentClassification] Loading model from: {self.model_path}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self.model_path),
            max_seq_length=m_cfg.get("max_seq_length", 2048),
            load_in_4bit=m_cfg.get("load_in_4bit", True),
        )
        FastLanguageModel.for_inference(self.model)
        self.device = self.model.device
        print(f"[IntentClassification] Model loaded successfully!")

        # ── Generation config ──────────────────────────────────────────────────
        g_cfg = self.cfg.get("generation", {})
        self.max_new_tokens = g_cfg.get("max_new_tokens", 64)
        self.temperature = g_cfg.get("temperature", 0.0)
        self.do_sample = g_cfg.get("do_sample", False)
        self.top_p = g_cfg.get("top_p", 1.0)
        self.use_cache = g_cfg.get("use_cache", True)
        self.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, message: str) -> str:
        """
        Dự đoán intent label cho một câu.

        Args:
            message: Câu hỏi / utterance đầu vào (str)

        Returns:
            predicted_label: Intent label dự đoán (str)
        """
        prompt = ALPACA_PROMPT.format(message)
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_p=self.top_p,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
        )

        # ── Extract generated text (bỏ phần prompt) ─────────────────────────
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return self._match_intent(generated_text)

    def _match_intent(self, generated: str) -> str:
        """Fuzzy match generated text với 77 intent labels."""
        generated = generated.strip()

        # Exact match
        if generated in self.id2label.values():
            return generated

        # Partial match
        for intent in self.id2label.values():
            if intent in generated or generated in intent:
                return intent

        return generated

    def batch_predict(self, messages: list[str]) -> list[str]:
        """Dự đoán intent cho nhiều câu cùng lúc."""
        return [self(msg) for msg in messages]


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Banking Intent Classification Inference")
    parser.add_argument("-m", "--model_path", type=str, required=True,
                        help="Thư mục chứa model checkpoint")
    parser.add_argument("-c", "--config", type=str, default="configs/inference.yaml",
                        help="File config inference YAML")
    parser.add_argument("-q", "--input_text", type=str, default=None,
                        help="Một câu hỏi duy nhất cần phân loại")
    parser.add_argument("-f", "--input_file", type=str, default=None,
                        help="File .txt chứa nhiều câu (mỗi dòng = 1 câu)")
    parser.add_argument("-o", "--output_file", type=str, default=None,
                        help="Lưu kết quả ra file JSON")
    parser.add_argument("--reload", action="store_true",
                        help="Force reload model (bỏ qua cache)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Banking Intent Classification — Inference")
    print("=" * 60)

    # ── Load model (cache) ─────────────────────────────────────────────────────
    classifier = get_cached_model(args.model_path, force_reload=args.reload)
    print("")

    # ── Parse queries ───────────────────────────────────────────────────────────
    queries = []
    if args.input_text:
        queries = [args.input_text]
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(queries)} queries from: {args.input_file}")

    # ── Run inference ──────────────────────────────────────────────────────────
    if not queries:
        # Interactive mode
        print("Interactive mode — Nhập câu hỏi banking (Ctrl+C để thoát)")
        print("-" * 60)
        while True:
            try:
                q = input("\nCâu hỏi > ").strip()
                if not q:
                    continue
                label = classifier(q)
                print(f"Intent    : {label}")
            except KeyboardInterrupt:
                print("\nThoát.")
                break
            except EOFError:
                break
    else:
        # Batch / single mode
        results = []
        for q in queries:
            label = classifier(q)
            results.append({
                "query": q,
                "predicted_intent": label,
            })
            print(f"Câu hỏi : {q}")
            print(f"Intent   : {label}")
            print("-" * 60)

        if args.output_file:
            Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nKết quả đã lưu vào: {args.output_file}")


if __name__ == "__main__":
    main()
