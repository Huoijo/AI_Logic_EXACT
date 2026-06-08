# EXACT Kaggle-Core xAI Starter

Dự án này thiết kế theo hướng **Kaggle làm GPU core chính** cho model open-source ≤8B, còn logic xAI nằm trong code Python có thể test local.

## Ý tưởng

```text
VSCode local
  -> git push
  -> ./run_kaggle.sh batch
  -> Kaggle chạy Qwen3-8B + symbolic reasoner
  -> tải artifacts/answers.json + eval_report.json về máy
```

Core reasoning không để LLM tự trả lời trực tiếp. LLM chỉ làm:

1. parse câu hỏi / MCQ option sang query logic;
2. hỗ trợ rewrite explanation;
3. fallback khi rule-based parser không hiểu.

Câu trả lời cuối cùng đi qua symbolic reasoner:

```text
prove(Q)    -> Yes
prove(not Q)-> No
else        -> Uncertain
```

## Cấu trúc

```text
exact_xai/
  api.py              # API local để test format
  schemas.py          # Pydantic models
  fol.py              # parser FOL đơn giản
  reasoner.py         # forward-chaining symbolic reasoner
  query_parser.py     # rule-based + LLM query parser
  llm_qwen.py         # Qwen loader/generator cho Kaggle
  pipeline.py         # end-to-end answer pipeline
  explanation.py      # proof -> explanation
  io_utils.py         # load/save JSON

scripts/
  kaggle_core_runner.py  # script chính chạy trên Kaggle
  local_eval.py          # chạy nhanh local không cần GPU
  make_request.py        # tạo input_requests.json mẫu

kaggle_job/
  runner.py              # Kaggle wrapper: clone repo, install, run script
  kernel-metadata.json   # cấu hình Kaggle kernel/script

run_kaggle.sh            # local one-command runner
requirements.txt         # local CPU requirements
requirements_kaggle.txt  # Kaggle GPU requirements
```

## Cài local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Test nhanh không GPU:

```bash
python scripts/local_eval.py --dataset data/fraction_dataset.json --limit 3
```

Chạy API local để test schema:

```bash
uvicorn exact_xai.api:app --reload --port 8080
```

## Setup Kaggle CLI

Tải `kaggle.json` từ Kaggle Account Settings, rồi đặt vào:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
pip install kaggle
```

Sửa 3 chỗ trước khi chạy:

1. `kaggle_job/kernel-metadata.json`: đổi `YOUR_KAGGLE_USERNAME`.
2. `kaggle_job/runner.py`: đổi `REPO_URL`.
3. `run_kaggle.sh`: đổi `KERNEL`.

Sau đó:

```bash
chmod +x run_kaggle.sh
./run_kaggle.sh batch
```

Output sẽ nằm ở:

```text
artifacts/answers.json
artifacts/eval_report.json
artifacts/bad_cases.json
```

## Kaggle core modes

```bash
./run_kaggle.sh batch     # chạy toàn dataset / request batch
./run_kaggle.sh eval      # tương tự batch, có report accuracy nếu có gold answer
./run_kaggle.sh smoke     # chỉ chạy vài sample để kiểm tra môi trường
```

## Ghi chú rule competition

- Model chính mặc định: `Qwen/Qwen3-8B`.
- Không dùng MoE >8B total params.
- Không chạy nhiều LLM song song.
- Nếu dùng model phụ, chỉ dùng tuần tự và thay `MODEL_NAME` trong runner.

## Public API ngày chấm

Kaggle không phù hợp làm persistent public API. Nếu ban tổ chức bắt gọi HTTP API trực tiếp, dùng `exact_xai.api` làm thin façade, còn inference nặng vẫn chạy batch trên Kaggle. Nếu họ chấp nhận input/output batch, `scripts/kaggle_core_runner.py` là core chính.

## v3 additions: NL-premise mode + benchmark

The original skeleton could consume `premises-FOL` directly. v3 adds the missing competition path:

```text
premises-NL + question
  -> NL-to-logic compiler
  -> validated FOL-like rules
  -> symbolic reasoner + optional Z3 backend
  -> answer + proof + cited explanation
```

### Local benchmark

Use FOL if available:

```bash
python scripts/kaggle_core_runner.py --task benchmark --dataset data/fraction_dataset.json --input-mode auto --out artifacts/local_auto
```

Force real spec behavior: ignore gold FOL and translate from NL:

```bash
python scripts/kaggle_core_runner.py --task batch_nl --dataset data/fraction_dataset.json --input-mode nl --out artifacts/local_nl --limit 10
```

Batch validation:

```bash
python scripts/kaggle_core_runner.py --task benchmark --dataset data/fraction_dataset.json --input-mode nl --batch-size 16 --out artifacts/bench_nl
```

Outputs:

```text
answers.json
answers.partial.json
bad_cases.json
warning_cases.json
batch_reports.json
eval_report.json
translation_report.json    # for --task translate
translations.json          # for --task translate
```

### Kaggle benchmark

```bash
./run_kaggle.sh smoke
./run_kaggle.sh batch
./run_kaggle.sh batch_nl
BATCH_SIZE=16 LIMIT=100 ./run_kaggle.sh benchmark
./run_kaggle.sh translate
```

### Answer order repair

The dataset loader now aligns gold answers by question type. If the JSON stores answers in the wrong order, for example `["Yes", "A"]` while question 0 is MCQ and question 1 is Yes/No, the loader swaps them internally and records a `loader_warnings` entry in `eval_report.json`.

### Z3 backend

`z3-solver` is optional but enabled by default when installed. It is used for finite-domain entailment checks, especially MCQ options that are implications or contraposition-like statements. The custom reasoner is still used first because it gives cleaner proof trees.
