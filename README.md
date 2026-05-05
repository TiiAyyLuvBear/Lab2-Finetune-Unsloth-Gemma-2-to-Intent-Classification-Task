# Banking Intent Classification with Unsloth + Gemma-2

Fine-tune **Gemma-2-9B** (QLoRA) cho bài toán phân loại 77 banking intents.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Training Configuration](#training-configuration)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Dataset](#dataset)
- [Troubleshooting](#troubleshooting)
- [Video Demo](#video-demo)

---

## Project Structure

```
.
├── train.sh                    # Training pipeline (data prep + fine-tune)
├── inference.sh                # Inference (single query, batch, interactive)
├── evaluate.sh                 # Evaluate accuracy + inference speed
├── requirements.txt             # Python dependencies
├── configs/
│   ├── train.yaml               # Training config (model, LoRA, hyperparams)
│   └── inference.yaml          # Inference config (generation, paths)
├── scripts/
│   ├── train.py                # Fine-tune script (Unsloth QLoRA)
│   ├── preprocess_data.py      # Download Banking77 dataset
│   ├── inference.py            # Unified inference (class + CLI)
│   └── evaluate_model.py       # Accuracy + speed evaluation
├── sample_data/                # Dữ liệu sau khi chạy preprocess
│   ├── train.csv
│   ├── test.csv
│   └── id2label.csv
├── outputs/
│   └── banking_intent_model/   # Model sau khi train
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer files
└── README.md
```

---



## Environment Setup

### Hardware Requirements

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|------------|-------------------|-------------|
| GPU VRAM   | ≥ 14 GB           | 16 GB (Tesla T4 / A10 / A100) |
| Số GPU     | 1                 | 1 (multi-GPU cần cấu hình thêm) |
| RAM         | 16 GB             | 32 GB |
| Disk       | 20 GB free        | 50 GB (model + dataset) |

### Software Requirements

| Thành phần | Phiên bản |
|------------|-----------|
| Python     | ≥ 3.10    |
| PyTorch    | ≥ 2.0.0   |
| CUDA       | ≥ 12.0    |
| cuDNN      | Compatible với CUDA version |

### CUDA Libraries (QUAN TRỌNG)

**Đối với Kaggle / Colab**, `bitsandbytes` yêu cầu các CUDA runtime libraries:

```
libnvJitLink.so.13    # JIT linking library (CUDA 13.x)
libnvrtc.so.X         # NVIDIA CUDA Runtime
libnvToolsExt.so.X   # NVIDIA Tools Extension
```

Kiểm tra xem các file này có trong hệ thống không:

```bash
# Tìm CUDA libraries trên máy
find /usr/local/cuda* -name "libnvJitLink*" 2>/dev/null
echo $LD_LIBRARY_PATH
```

### Installation

```bash
# 1. Clone repository
git clone <repo_url>
cd Lab2-Finetune-Unsloth-Gemma-2-to-Intent-Classification-Task

# 2. Install dependencies
pip install -r requirements.txt

# 3. CUDA environment variables (THÊM VÀO ~/.bashrc ĐỂ PERSIST)
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
```

### Kaggle Notebook Setup

Nếu chạy trên Kaggle, các script đã tự động set `CUDA_VISIBLE_DEVICES` và `LD_LIBRARY_PATH`:

- **`train.sh`**: Tự động tìm CUDA path và export `LD_LIBRARY_PATH`
- **`evaluate.sh`**: Force single-GPU (`CUDA_VISIBLE_DEVICES=0`) tránh cross-device crash
- **`inference.py`**: Force single-GPU (`CUDA_VISIBLE_DEVICES=0`)

---

## Training Configuration

### Model Configuration (`configs/train.yaml`)

```yaml
# ── Model ─────────────────────────────────────────────────────────────────────
model:
  name: "unsloth/gemma-2-9b-bnb-4bit"    # Gemma-2-9B, 4-bit quantized
  max_seq_length: 2048                     # Maximum sequence length
  dtype: null                              # null = auto, float16 = T4/V100, bfloat16 = Ampere+
  load_in_4bit: true                        # QLoRA: 4-bit quantization

# ── LoRA (Low-Rank Adaptation) ───────────────────────────────────────────────
lora:
  r: 16                                     # Rank: 8 (fast) → 64 (quality)
  lora_alpha: 16                            # Scaling factor (thường = r)
  lora_dropout: 0.0                         # 0.0 = tối ưu VRAM
  bias: "none"                             # "none" = tối ưu, không train bias
  gradient_checkpointing: "unsloth"        # Giảm 30% VRAM, cho phép batch lớn hơn
  target_modules:                           # Các layers được apply LoRA
    - "q_proj"     # Query projection
    - "k_proj"     # Key projection
    - "v_proj"     # Value projection
    - "o_proj"     # Output projection
    - "gate_proj"  # Feed-forward gate
    - "up_proj"    # Feed-forward up
    - "down_proj"  # Feed-forward down

# ── Training Hyperparameters ─────────────────────────────────────────────────
training:
  per_device_train_batch_size: 2           # Batch size trên mỗi GPU
  gradient_accumulation_steps: 4           # Effective batch = 2 × 4 = 8
  learning_rate: 2e-4                      # LR cho QLoRA (thường 1e-4 → 3e-4)
  num_train_epochs: 3                       # Số epoch huấn luyện
  max_steps: -1                             # -1 = train đủ num_train_epochs
  warmup_steps: 5                           # Warmup steps (overfit prevention)
  logging_steps: 20                         # Log sau mỗi N steps
  optim: "adamw_8bit"                       # Memory-efficient optimizer (Unsloth)
  weight_decay: 0.001                       # Weight decay (regularization)
  lr_scheduler_type: "linear"              # Linear decay LR
  seed: 3407                                # Reproducibility seed
  packing: false                            # false = mỗi sequence độc lập

# ── Output ────────────────────────────────────────────────────────────────────
output:
  dir: "outputs"
  save_merged: false                       # true = lưu merged model (full model)
  report_to: "none"                         # "none", "wandb", "tensorboard"
```

### Configuration Parameters Explained

| Parameter | Giá trị mặc định | Ý nghĩa |
|-----------|------------------|---------|
| `model.name` | `unsloth/gemma-2-9b-bnb-4bit` | Base model (Gemma-2-9B, 4-bit BNB quantized) |
| `model.max_seq_length` | `2048` | Độ dài token tối đa cho mỗi input |
| `model.load_in_4bit` | `true` | Bật QLoRA 4-bit quantization |
| `lora.r` | `16` | LoRA rank — càng cao càng nhiều parameters trainable (0.58% mặc định) |
| `lora.gradient_checkpointing` | `unsloth` | Kỹ thuật tiết kiệm VRAM |
| `training.per_device_train_batch_size` | `2` | Số samples mỗi GPU mỗi step |
| `training.gradient_accumulation_steps` | `4` | Tích lũy gradient → effective batch = 8 |
| `training.learning_rate` | `2e-4` | Learning rate cho LoRA adapters |
| `training.num_train_epochs` | `3` | Số lần duyệt qua toàn bộ dataset |

### Effective Training Metrics

Với cấu hình mặc định trên **1 GPU Tesla T4 (14.6 GB)**:

```
Num examples       = 1,000
Num Epochs         = 3
Total steps        = 375
Batch per GPU      = 2
Grad accumulation  = 4
Total batch size   = 8
Trainable params   = 54,018,048 / 9,295,724,032 (0.58%)
```

---

## Training

### Quick Start

```bash
# Chạy đầy đủ (prepare data → train → save model)
bash train.sh

# Resume training (bỏ qua data prep, tiếp tục từ checkpoint)
bash train.sh -r
```

### Custom Options

```bash
bash train.sh -d sample_data -c configs/train.yaml -o outputs -s 3000
```

| Flag | Mặc định | Mô tả |
|------|----------|-------|
| `-d` | `sample_data` | Thư mục chứa dữ liệu |
| `-c` | `configs/train.yaml` | File cấu hình training |
| `-o` | `outputs` | Thư mục lưu checkpoint |
| `-s` | `3000` | Số mẫu train (sampling từ 9993) |
| `-r` | false | Resume từ checkpoint đã lưu |
| `-h` | — | Hiển thị help |

### Training Pipeline Steps

```
=== [1/2] Chuẩn bị dữ liệu ===
  └─ Download Banking77 từ mteb
  └─ Tạo sample_data/train.csv, test.csv, id2label.csv
  └─ Sampling 1000 mẫu train (mặc định)

=== [2/2] Fine-tune model ===
  └─ Load Gemma-2-9B (4-bit QLoRA)
  └─ Attach LoRA adapters (r=16, 0.58% params trainable)
  └─ Format dataset theo Alpaca prompt template
  └─ Train với SFT Trainer
  └─ Save adapters tại outputs/banking_intent_model/
```

### Expected Output After Training

```
outputs/
└── banking_intent_model/
    ├── adapter_config.json      # LoRA adapter config
    ├── adapter_model.safetensors # Trained LoRA weights
    ├── model.safetensors        # (nếu save_merged=true)
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── special_tokens_map.json
```

---

## Inference

### Python API

```python
from scripts.inference import IntentClassification

classifier = IntentClassification("outputs/banking_intent_model")
label = classifier("I want to transfer money to another account")
print(label)  # → transfer_into_account
```

### CLI

```bash
# Interactive mode
bash inference.sh -m outputs/banking_intent_model

# Single query
bash inference.sh -m outputs/banking_intent_model -q "I lost my card"

# Batch từ file
bash inference.sh -m outputs/banking_intent_model -f queries.txt -o results.json
```

| Flag | Mô tả |
|------|-------|
| `-m` | Đường dẫn model checkpoint |
| `-q` | Câu hỏi duy nhất |
| `-f` | File chứa nhiều câu (mỗi dòng = 1 câu) |
| `-o` | Lưu kết quả ra JSON |

---

## Evaluation

```bash
# Đánh giá đầy đủ trên tập test
bash evaluate.sh

# Giới hạn 500 mẫu + hiển thị chi tiết
bash evaluate.sh -n 500 -d

# Lưu kết quả ra JSON
bash evaluate.sh -o results.json
```

| Flag | Mặc định | Mô tả |
|------|----------|-------|
| `-m` | `outputs/banking_intent_model` | Đường dẫn model |
| `-n` | `1000` | Số mẫu test |
| `-d` | false | Hiển thị chi tiết từng prediction |
| `-o` | — | Lưu kết quả ra JSON |

### Metrics

| Metric | Ý nghĩa |
|--------|---------|
| `accuracy` | Độ chính xác (%) trên tập test |
| `avg_latency_ms` | Tốc độ trung bình (ms/câu) |
| `max_latency_ms` | Tốc độ max |
| `min_latency_ms` | Tốc độ min |

---

## Dataset

- **Banking77** từ `mteb/banking77`
- **77 intent classes** — 77 loại ý định ngân hàng khác nhau
- **~13,000 samples** — 9,993 train, 3,076 test
- Tự động download khi chạy `train.sh` hoặc `python scripts/preprocess_data.py`

```bash
# Tùy chỉnh số mẫu train
python scripts/preprocess_data.py --data_dir sample_data --train_samples 3000
```

---

## Troubleshooting

---

### 🚨 Lỗi `found two different devices cuda:0, cuda:1`

```
TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
  got RuntimeError('Unhandled FakeTensor Device Propagation for aten.add_.Tensor,
  found two different devices cuda:1, cuda:0')
```

**Nguyên nhân:** Kaggle có 2 GPU Tesla T4 nhưng Unsloth phân tán model/tensor sang nhiều GPU → attention kernel chạy cross-device (`cuda:0` vs `cuda:1`) → crash.

**Cách sửa**

Set `CUDA_VISIBLE_DEVICES=0` trong môi trường để force sử dụng GPU đầu tiên:

```bash
# evaluate.sh đã tự động set:
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
```

```python
# evaluate_model.py và inference.py đã tự động set:
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```


---

### 🚨 Lỗi `CUDA out of memory`

```
RuntimeError: CUDA out of memory. Tried to allocate 16.00 GiB
```

**Nguyên nhân:** Batch size quá lớn cho GPU hiện tại.

**Cách sửa — Giảm batch size trong `configs/train.yaml`:**

```yaml
training:
  per_device_train_batch_size: 1    # Giảm từ 2 → 1
  gradient_accumulation_steps: 8     # Tăng để giữ effective batch = 8
```

**Hoặc tăng gradient checkpointing:**

```yaml
lora:
  gradient_checkpointing: "unsloth"  # Đảm bảo đang bật
```

---

### 🚨 Lỗi `FileNotFoundError: id2label.csv`

```
FileNotFoundError: Không tìm thấy id2label.csv tại: sample_data/id2label.csv
```

**Nguyên nhân:** Chưa chạy `preprocess_data.py`.

**Cách sửa:**

```bash
python scripts/preprocess_data.py --data_dir sample_data --train_samples 1000
```

---

### 🚨 Lỗi `FileNotFoundError: banking_intent_model`

```
FileNotFoundError: Model path outputs/banking_intent_model does not exist
```

**Nguyên nhân:** Chưa train model hoặc thư mục `outputs/` trống.

**Cách sửa:**

```bash
bash train.sh
```

---

### 🚨 Lỗi `Unsloth is not installed`

```
RuntimeError: Unsloth is required. Cài đặt: pip install unsloth
```

**Cách sửa:**

```bash
pip install unsloth
```

Hoặc cài phiên bản mới nhất:

```bash
pip install --upgrade unsloth
```

---

### Kiểm tra GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## Model Caching

Model được **cache trong memory** sau lần load đầu tiên — không reload mỗi lần gọi:

- ✅ **CACHE HIT**: Gọi nhiều lần trong cùng Python process
- ✅ **Interactive mode**: Tất cả queries dùng chung model đã load
- ⚠️ Process mới: Mỗi script chạy riêng (Python process isolation)

---

## Requirements

```
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
# transformers>=4.51.3,<5.0.0
# trl>=0.18.2,<0.24.0
transformers==4.53.2
trl==0.20.0
datasets
pandas
pyyaml
scikit-learn
```

## Video Demo


- Google Drive (public): <https://drive.google.com/file/d/1amOKIxbHOPyUvCg0fdTUacVmSCssorF0/view?usp=sharing>
