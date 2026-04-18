# Banking Intent Classification with Unsloth + Gemma-2

Fine-tune **Gemma-2-9B** (QLoRA) cho bài toán phân loại 77 banking intents.

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
└── README.md
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Tải model (chạy train.sh sẽ tự động tải)
# Sau khi train xong, model được lưu tại:
#   outputs/banking_intent_model/
```

---

## Training

```bash
# Chạy đầy đủ (prepare data → train → save model)
bash train.sh

# Tùy chỉnh tham số
bash train.sh -d sample_data -c configs/train.yaml -o outputs -s 3000

# Resume training (bỏ qua data prep, tiếp tục từ checkpoint)
bash train.sh -r
```

### Training Options

| Flag | Mặc định | Mô tả |
|------|----------|--------|
| `-d` | `sample_data` | Thư mục chứa data |
| `-c` | `configs/train.yaml` | File config |
| `-o` | `outputs` | Thư mục lưu checkpoint |
| `-s` | `3000` | Số mẫu train |
| `-r` | false | Resume từ checkpoint |
| `-h` | — | Hiển thị help |

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

# Sử dụng trực tiếp với venv python
./bin/python3 scripts/inference.py -m outputs/banking_intent_model -q "Refund my transaction"
```

### Inference Options

| Flag | Mô tả |
|------|-------|
| `-m` | Đường dẫn model checkpoint (mặc định: `outputs/banking_intent_model`) |
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

### Evaluation Options

| Flag | Mặc định | Mô tả |
|------|----------|--------|
| `-m` | `outputs/banking_intent_model` | Đường dẫn model |
| `-n` | `1000` | Số mẫu test |
| `-d` | false | Hiển thị chi tiết từng prediction |
| `-o` | — | Lưu kết quả ra JSON |

### Metrics

| Metric | Ý nghĩa |
|--------|---------|
| `accuracy` | Độ chính xác (%) trên tập test |
| `avg_latency_ms` | Tốc độ truy xuất TB (ms/câu) |
| `max_latency_ms` | Tốc độ max |
| `min_latency_ms` | Tốc độ min |

---

## Model Caching

Model được **cache trong memory** sau lần load đầu tiên — không reload mỗi lần gọi:

- ✅ **CACHE HIT**: Gọi nhiều lần trong cùng Python process
- ✅ **Interactive mode**: Tất cả queries dùng chung model đã load
- ⚠️ Process mới: Mỗi script chạy riêng (Python process isolation)

---

## Dataset

- **Banking77** từ `mteb/banking77`
- 77 intent classes, ~13K samples
- Tự động download khi chạy `train.sh`

---

## Requirements

```
torch>=2.0.0
transformers==4.56.2
datasets==4.3.0
accelerate>=0.25.0
peft>=0.6.0
trl==0.22.2
unsloth
sentencepiece, protobuf
pandas, pyyaml, tqdm
bitsandbytes>=0.41.0, hf_transfer
```
