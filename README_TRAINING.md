# EXACT NL→FOL QLoRA Workflow

This workflow trains an **NL→FOL/IR compiler**, not an answer predictor.

Target model by default:

```bash
Qwen/Qwen2.5-7B-Instruct
```

The final answer should still be produced by the symbolic reasoner/Z3 pipeline.

## 0. Prepare full dataset

Put the full dataset in:

```bash
data/full_data.json
```

Check counts:

```bash
python scripts/count_dataset_questions.py --dataset data/full_data.json
```

## 1. Build SFT data locally

```bash
python scripts/build_sft_data.py \
  --dataset data/full_data.json \
  --out-dir data/sft_nl2fol \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --seed 42
```

Outputs:

```text
data/sft_nl2fol/
  train.jsonl
  valid.jsonl
  test.jsonl
  split_report.json
  noisy_cases.json
```

Training samples include:

- `premise_translation`: one NL premise → one FOL premise, gold.
- `record_premise_compiler`: all NL premises → all FOL premises, gold.
- `question_intent`: question → type/intent/options text, derived.
- `silver_option_compiler`: MCQ options → conservative silver FOL when regex can parse.
- `full_compiler`: whole record → logic IR, gold premises and optional silver options.

## 2. Train on Kaggle

Small smoke training first:

```bash
MAX_TRAIN_SAMPLES=50 MAX_VALID_SAMPLES=20 EPOCHS=1 \
DATASET=data/full_data.json \
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
./run_train_kaggle.sh
```

Real first run:

```bash
DATASET=data/full_data.json \
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
EPOCHS=2 \
MAX_SEQ_LENGTH=2048 \
TRAIN_BS=1 \
GRAD_ACCUM=8 \
./run_train_kaggle.sh
```

Output after download:

```text
train_artifacts/
  adapter/
    adapter_config.json
    adapter_model.safetensors
    train_report.json
  sft_data/
    split_report.json
    noisy_cases.json
```

## 3. Use adapter for evaluation

Copy or keep adapter path, then run benchmark with base model + adapter:

```bash
INPUT_MODE=nl \
DATASET=data/full_data.json \
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
ADAPTER_PATH="train_artifacts/adapter" \
LIMIT=50 BATCH_SIZE=8 LOG_EACH_CASE=1 \
./run_kaggle.sh benchmark
```

## 4. Important rule

Do **not** train the model to output final answer labels like A/Yes/No. The model is only a compiler:

```text
NL premises + question/options → JSON/FOL/IR
```

Then:

```text
JSON/FOL/IR → validator → symbolic reasoner/Z3 → answer + proof
```
