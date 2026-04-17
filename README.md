# banking-intent-unsloth

Fine-tuning a **banking intent classifier** using [Unsloth](https://github.com/unslothai/unsloth) + QLoRA.
Supports LLaMA-3, Mistral, Phi-3 and other popular open LLMs with 4-bit quantisation and LoRA adapters.

---

## Project structure

```
banking-intent-unsloth/
├── scripts/
│   ├── preprocess_data.py   # Tokenise & format CSV → HuggingFace Dataset
│   ├── train.py             # Fine-tune with Unsloth + LoRA
│   └── inference.py         # Run predictions (single / batch / interactive)
├── configs/
│   ├── train.yaml           # Training hyperparameters
│   └── inference.yaml       # Inference / generation settings
├── sample_data/
│   ├── train.csv            # Labelled training examples
│   ├── test.csv             # Held-out evaluation examples
│   └── processed/           # Auto-created after preprocessing
├── train.sh                 # Run preprocessing + training end-to-end
├── inference.sh             # Run inference
└── requirements.txt
```

---

## Intent labels

| Label | Description |
|---|---|
| `balance_inquiry` | User asks about account balance |
| `transfer` | User wants to move money |
| `card_block` | User reports lost/stolen card |
| `loan_inquiry` | User asks about loan products |
| `complaint` | User files a service complaint |
| `unknown` | Query doesn't match any intent |

---

## Quick start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Unsloth currently requires CUDA 12.1+. See the [Unsloth install guide](https://github.com/unslothai/unsloth) for GPU-specific instructions.

### 2 — Preprocess data

```bash
python scripts/preprocess_data.py \
    --data_dir sample_data \
    --output_dir sample_data/processed \
    --model_name "unsloth/llama-3-8b-bnb-4bit" \
    --max_length 128
```

### 3 — Train

```bash
bash train.sh
# or step-by-step:
python scripts/train.py \
    --config configs/train.yaml \
    --data_dir sample_data/processed/train \
    --output_dir outputs
```

### 4 — Run inference

```bash
# Single query
bash inference.sh outputs/final

# From file (one query per line)
python scripts/inference.py \
    --model_path outputs/final \
    --input_file queries.txt \
    --output_file results.json

# Interactive mode
python scripts/inference.py --model_path outputs/final
```

---

## Configuration

All hyperparameters are managed in `configs/train.yaml` / `configs/inference.yaml`.
Key knobs:

| Parameter | Default | Description |
|---|---|---|
| `model.name` | `unsloth/llama-3-8b-bnb-4bit` | Base model |
| `training.lora_r` | `16` | LoRA rank |
| `training.learning_rate` | `2e-4` | Peak learning rate |
| `training.num_train_epochs` | `3` | Training epochs |
| `training.per_device_train_batch_size` | `2` | Per-device batch size |
| `training.gradient_accumulation_steps` | `4` | Effective batch = 2 × 4 = 8 |

---

## Extending

- **Add new intents** — append to `LABEL_MAP` in `preprocess_data.py` and `LABEL_LIST` in `inference.py`.
- **Swap base model** — change `model.name` in `configs/train.yaml` (e.g. `unsloth/mistral-7b-bnb-4bit`).
- **Add experiment tracking** — set `report_to: wandb` in `train.yaml` and run `wandb login` first.
- **Export to GGUF** — Unsloth supports `model.save_pretrained_gguf()` for llama.cpp deployment.

---

## License

MIT
