"""
preprocess_data.py
Preprocesses raw CSV data into instruction-tuned format for intent classification.
Loads train.csv / test.csv from sample_data/, formats them as instruction prompts,
and saves tokenized datasets ready for Unsloth fine-tuning.
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── prompt templates ──────────────────────────────────────────────────────────
INTENT_TEMPLATE = "Classify the banking intent of the following query.\nQuery: {query}\nIntent: "
LABEL_MAP = {
    "balance_inquiry": 0,
    "transfer": 1,
    "card_block": 2,
    "loan_inquiry": 3,
    "complaint": 4,
    "unknown": 5,
}


def format_example(row: pd.Series) -> dict:
    """Convert a DataFrame row into an instruction-tuned example dict."""
    return {
        "text": INTENT_TEMPLATE.format(query=row["text"]),
        "label": LABEL_MAP.get(row["label"].strip().lower(), LABEL_MAP["unknown"]),
    }


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int = 128) -> Dataset:
    """Tokenize text fields and add model-ready input/label columns."""
    def _tokenize(example):
        tokenized = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = example["label"]
        return tokenized

    return dataset.map(_tokenize, remove_columns=["text"])


def load_and_format(csv_path: str) -> Dataset:
    """Load a CSV file and return a HuggingFace Dataset with required columns."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}  (found: {list(df.columns)})")

    examples = [format_example(row) for _, row in df.iterrows()]
    formatted_df = pd.DataFrame(examples)
    return Dataset.from_pandas(formatted_df)


def main():
    parser = argparse.ArgumentParser(description="Preprocess banking intent data for Unsloth training.")
    parser.add_argument("--data_dir", type=str, default="sample_data", help="Directory containing train.csv and test.csv")
    parser.add_argument("--output_dir", type=str, default="sample_data/processed", help="Directory to save processed datasets")
    parser.add_argument("--model_name", type=str, default="unsloth/llama-3-8b-bnb-4bit", help="Tokenizer model name or path")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for split in ["train", "test"]:
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            logger.warning("Skipping %s split — file not found: %s", split, csv_path)
            continue

        logger.info("Processing %s split from %s", split, csv_path)
        dataset = load_and_format(str(csv_path))
        dataset = tokenize_dataset(dataset, tokenizer, max_length=args.max_length)
        dataset.save_to_disk(str(output_dir / split))
        logger.info("Saved processed %s split → %s", split, output_dir / split)

    logger.info("Preprocessing complete. Processed datasets saved to %s", output_dir)


if __name__ == "__main__":
    main()
