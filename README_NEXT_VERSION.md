# EXACT NL2FOL Workflow - Next Patch

This patch targets the failure mode seen after the first adapter-backed benchmark:

- Yes/No questions were sometimes parsed as fake multiple choice questions.
- MCQ options that are factual statements about named entities, e.g. `Sophia qualifies...`, were sometimes parsed as universal rules `ForAll(...)`.
- This made material implication too permissive and caused `multiple_provable_options` false positives.

## Main changes

### `exact_xai/query_parser.py`

Adds deterministic post-processing after LLM parsing:

- If the original question has no A/B/C/D options, force `kind=yes_no`.
- If an MCQ option is not an explicit `If ..., then ...` statement, do not allow `ForAll(...)`.
- If an MCQ option mentions a named entity from the KB, force an atom target like `eligible_international_program(Sophia)`.
- Explicit conditional options can still become `ForAll(x, A(x) -> B(x))`.

Expected fix for Sophia case:

```json
{
  "A": "qualifies_university_scholarship(Sophia)",
  "B": "received_faculty_recommendation(Sophia)",
  "C": "eligible_international_program(Sophia)",
  "D": "passed_language_proficiency_exam(Sophia)"
}
```

### `exact_xai/pipeline.py`

- Calls `postprocess_parsed_question(...)` after LLM query parsing.
- Adds parser postprocess warnings to the response for debugging.
- Scores multiple provable MCQ options by `(used premise count, proof length, label)` instead of proof length only.

### `scripts/train_lora_nl2fol.py`

- Uses prepared `SFT_DIR/train.jsonl` and `SFT_DIR/valid.jsonl` directly.
- Keeps compatibility with multiple TRL versions by filtering unsupported kwargs.
- Writes both `train_report.json` and `training_meta.json`.

## Suggested validation command

```bash
ADAPTER_PATH="train_artifacts/adapter" \
INPUT_MODE=nl \
DATASET=data/full_data.json \
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
LIMIT=3 BATCH_SIZE=1 LOG_EACH_CASE=1 \
./run_kaggle.sh benchmark
```

Goal:

```text
Case 0:0 -> A
Case 0:1 -> Yes
Case 1:0 -> C
```
