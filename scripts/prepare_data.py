"""
prepare_data.py
Tải bộ Banking77 từ mteb, tách thành train / test CSV,
lưu id2label mapping cho inference và evaluation.
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Tải và chuẩn bị dữ liệu Banking77.")
    parser.add_argument("--data_dir", type=str, default="sample_data", help="Thư mục lưu dữ liệu")
    parser.add_argument("--train_samples", type=int, default=None, help="Số mẫu train giữ lại (sampling)")
    parser.add_argument("--seed", type=int, default=42, help="Seed cho random sampling")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Tải Banking77 ────────────────────────────────────────────────────────
    logger.info("Đang tải bộ dữ liệu mteb/banking77 ...")
    dataset = load_dataset("mteb/banking77")
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    logger.info("Train: %d | Test: %d", len(train_df), len(test_df))

    # ── 2. Lưu id2label mapping (đủ 77 intent, không sampling) ───────────────────
    id2label = dict(zip(train_df["label"].astype(str), train_df["label_text"]))
    mapping_df = pd.DataFrame(list(id2label.items()), columns=["label", "label_text"])
    mapping_path = data_dir / "id2label.csv"
    mapping_df.to_csv(mapping_path, index=False)
    logger.info("Đã lưu %d intent labels vào %s", len(id2label), mapping_path)

    # ── 3. Lưu train (sampling) & test CSV ─────────────────────────────────────
    if args.train_samples and len(train_df) > args.train_samples:
        logger.info("Sampling %d mẫu train (từ %d) ...", args.train_samples, len(train_df))
        train_df = train_df.sample(n=args.train_samples, random_state=args.seed)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Đã lưu train.csv (%d mẫu) và test.csv (%d mẫu)", len(train_df), len(test_df))

    logger.info("Hoàn tất chuẩn bị dữ liệu tại: %s", data_dir.resolve())


if __name__ == "__main__":
    main()