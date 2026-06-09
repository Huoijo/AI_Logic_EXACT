# EXACT v4 Patch — IR Validator + Repair + Requirement/Numeric Reasoning

This patch is designed after the `LIMIT=50` QA report where the system reached `13/50` with many warnings. The failures were concentrated in a few repeatable classes:

1. LLM-produced invalid FOL such as `John completed_courses()`, `x.completed_course_a`, and raw comparisons.
2. Numeric threshold reasoning such as `600 clinical hours >= 500 clinical hours`.
3. Negative/blocked requirements such as `not received_safety_endorsement(John)` blocking hazardous material transport.
4. MCQ options such as “needs more publications” being mapped to a positive fact.
5. Yes/No questions about “meet all requirements” returning `Uncertain` instead of `No` when a required condition is missing.

## New files

```text
exact_xai/fol_repair.py
exact_xai/requirement_reasoner.py
tests/test_v4_repair.py
```

## Patched files

```text
exact_xai/fol.py
exact_xai/reasoner.py
exact_xai/nl2logic.py
exact_xai/query_parser.py
exact_xai/pipeline.py
```

## What changed

### 1. FOL repair/validation

`fol_repair.py` repairs common invalid model outputs before parsing:

```text
John.completed_course_a                 -> completed_course_a(John)
x.enrolled_in_b                         -> enrolled_in_b(x)
John completes_thesis()                 -> completes_thesis(John)
Alex.membership_duration = 8_months     -> membership_duration(Alex, 8_months)
study_hours(x) >= 15                    -> study_hours_at_least_15(x)
x_can_apply_for_collaborative...        -> can_apply_collaborative_projects(...)
```

The NL2FOL translator now repairs before validation and warns with `premise_N_fol_repaired`.

### 2. Conjunction facts

`fol.py` now parses top-level conjunction facts:

```text
is_department_head(John) & has_degree(John, PhD)
```

as two facts instead of a single broken atom.

### 3. Numeric threshold matching

`reasoner.py` can match numeric-threshold predicates:

```text
completing_600_clinical_hours(John)
```

satisfies an antecedent requiring:

```text
completing_500_clinical_hours(x)
```

This targets cases like Nurse John, membership duration, GPA/threshold-style predicates, and course-count requirements.

### 4. Blocked negative requirement reasoning

`not_received_safety_endorsement(John)` is parsed as:

```text
not received_safety_endorsement(John)
```

If a downstream conclusion requires `received_safety_endorsement(John)`, the reasoner can now answer `No` instead of `Uncertain`.

### 5. Requirement-gap No reasoning

For questions containing wording like:

```text
meet all requirements
current eligibility status
```

`pipeline.py` calls `requirement_reasoner.py`. If the target has a direct rule but at least one required antecedent is missing or blocked, it returns `No` with warning `requirement_gap_no`.

### 6. MCQ post-processing improvements

`query_parser.py` now uses the entity from the question as context for pronoun/entity-free options like:

```text
He can propose new courses
Eligible for internship program
```

It also treats:

```text
needs more/additional/longer X
```

as a missing requirement, so it becomes a negated requirement query instead of a positive fact.

## How to apply

Copy/merge the patch folder into your repo, then run:

```bash
python -m py_compile exact_xai/fol_repair.py exact_xai/requirement_reasoner.py
python -m py_compile exact_xai/fol.py exact_xai/reasoner.py exact_xai/nl2logic.py exact_xai/query_parser.py exact_xai/pipeline.py
PYTHONPATH=. pytest -q tests/test_v4_repair.py
```

Commit:

```bash
git add exact_xai/fol_repair.py exact_xai/requirement_reasoner.py tests/test_v4_repair.py \
  exact_xai/fol.py exact_xai/reasoner.py exact_xai/nl2logic.py exact_xai/query_parser.py exact_xai/pipeline.py README_V4_PATCH.md

git commit -m "apply v4 fol repair and requirement reasoning patch"
git push
```

## Recommended benchmark ladder

```bash
ADAPTER_PATH="train_artifacts/adapter" INPUT_MODE=nl DATASET=data/full_data.json \
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" LIMIT=8 BATCH_SIZE=1 LOG_EACH_CASE=1 ./run_kaggle.sh benchmark

ADAPTER_PATH="train_artifacts/adapter" INPUT_MODE=nl DATASET=data/full_data.json \
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" LIMIT=16 BATCH_SIZE=1 LOG_EACH_CASE=1 ./run_kaggle.sh benchmark

ADAPTER_PATH="train_artifacts/adapter" INPUT_MODE=nl DATASET=data/full_data.json \
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" LIMIT=25 BATCH_SIZE=1 LOG_EACH_CASE=1 ./run_kaggle.sh benchmark
```

Do not jump to full benchmark immediately. First confirm that the earlier failure groups improve:

```text
Case 3: John fellowship
Case 4/5: Professor John needs-more/shallow-fact trap
Case 6: Hazmat No
Case 10: David internship syntax repair
Case 12/15: numeric threshold cases
```
