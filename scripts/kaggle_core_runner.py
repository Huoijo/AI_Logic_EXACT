from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from collections import Counter, defaultdict
from tqdm import tqdm

from exact_xai.io_utils import load_json, save_json
from exact_xai.schemas import AnswerRequest
from exact_xai.pipeline import AnswerPipeline
from exact_xai.llm_qwen import maybe_load_qwen
from exact_xai.fol import parse_fol_premises, parse_atom
from exact_xai.reasoner import Reasoner
from exact_xai.explanation import proof_to_explanation
from exact_xai.dataset_utils import records_from_exact_dataset, infer_question_kind


def records_from_batch(path: str):
    obj = load_json(path)
    for r in obj.get("records", []):
        yield AnswerRequest(**r)


def run_unit_smoke(out: Path):
    kb = parse_fol_premises([
        "ForAll(x, A(x) -> B(x))",
        "ForAll(x, B(x) -> C(x))",
        "A(Sophia)",
    ])
    rr = Reasoner(kb).prove_atom(parse_atom("C(Sophia)"))
    explanation = proof_to_explanation(rr.answer, rr.proof, explanation_style="short")
    row = {
        "id": "smoke:0",
        "answer": rr.answer,
        "target": "C(Sophia)",
        "proof": [step.model_dump() for step in rr.proof],
        "explanation": explanation,
        "gold_answer": "Yes",
    }
    report = {
        "task": "smoke",
        "num_records": 1,
        "num_with_gold": 1,
        "exact_match": 1.0 if rr.answer == "Yes" else 0.0,
        "model_mode": "symbolic_unit_smoke_no_llm",
    }
    save_json([row], out / "answers.json")
    save_json(report, out / "eval_report.json")
    save_json([], out / "bad_cases.json")
    save_json([], out / "warning_cases.json")
    print(report)


def _is_correct(pred, gold) -> bool:
    if gold is None:
        return False
    return str(pred).strip().lower() == str(gold).strip().lower()


def run_answer_batch(records: list[AnswerRequest], pipe: AnswerPipeline, out: Path, task: str, batch_size: int | None = None, loader_warnings: list[str] | None = None):
    results = []
    bad_cases = []
    warning_cases = []
    metrics_by_type = defaultdict(lambda: {"total": 0, "with_gold": 0, "correct": 0})
    warning_counter = Counter()
    parse_counter = Counter()

    correct = total_with_gold = 0
    if batch_size is None or batch_size <= 0:
        batch_size = len(records) or 1

    batch_reports = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        for req in tqdm(batch, desc=f"EXACT {task} {start}:{start+len(batch)}"):
            ans = pipe.answer(req)
            row = ans.model_dump()
            row["gold_answer"] = req.gold_answer
            row["question_type"] = req.question_type or infer_question_kind(req.question)
            results.append(row)

            qtype = row["question_type"]
            metrics_by_type[qtype]["total"] += 1
            if req.gold_answer is not None:
                total_with_gold += 1
                metrics_by_type[qtype]["with_gold"] += 1
                if _is_correct(ans.answer, req.gold_answer):
                    correct += 1
                    metrics_by_type[qtype]["correct"] += 1
                else:
                    bad_cases.append(row)

            if row.get("warnings"):
                warning_cases.append(row)
                for w in row["warnings"]:
                    warning_counter[w.split(":")[0]] += 1
            if ans.parsed_question:
                parse_counter[ans.parsed_question.parser] += 1

        batch_report = {
            "start": start,
            "end": start + len(batch),
            "num_records_done": len(results),
            "exact_match_so_far": correct / total_with_gold if total_with_gold else None,
            "num_bad_cases_so_far": len(bad_cases),
            "num_warning_cases_so_far": len(warning_cases),
        }
        batch_reports.append(batch_report)
        save_json(batch_report, out / "batch_progress" / f"batch_{start:06d}.json")
        save_json(results, out / "answers.partial.json")

    by_type = {}
    for k, v in metrics_by_type.items():
        by_type[k] = {
            **v,
            "exact_match": v["correct"] / v["with_gold"] if v["with_gold"] else None,
        }

    report = {
        "task": task,
        "num_records": len(records),
        "num_with_gold": total_with_gold,
        "exact_match": correct / total_with_gold if total_with_gold else None,
        "by_question_type": by_type,
        "num_bad_cases": len(bad_cases),
        "num_warning_cases": len(warning_cases),
        "warning_counter": dict(warning_counter),
        "parser_counter": dict(parse_counter),
        "loader_warnings": loader_warnings or [],
        "model_mode": pipe.answer(records[0]).mode if records else "none",
    }

    save_json(results, out / "answers.json")
    save_json(report, out / "eval_report.json")
    save_json(bad_cases, out / "bad_cases.json")
    save_json(warning_cases, out / "warning_cases.json")
    save_json(batch_reports, out / "batch_reports.json")
    print(report)


def run_translation_benchmark(records: list[AnswerRequest], pipe: AnswerPipeline, out: Path):
    rows = []
    ok = 0
    total = 0
    for req in tqdm(records, desc="NL->Logic benchmark"):
        kb, warnings, raw = pipe.build_kb(req)
        generated = raw.get("generated_premises_fol", req.premises_fol)
        total += 1
        valid = bool(generated)
        ok += int(valid)
        rows.append({
            "id": req.id,
            "valid": valid,
            "warnings": warnings,
            "generated_premises_fol": generated,
            "gold_premises_fol": req.premises_fol,
        })
    report = {
        "task": "translate",
        "num_records": total,
        "translation_nonempty_rate": ok / total if total else None,
    }
    save_json(rows, out / "translations.json")
    save_json(report, out / "translation_report.json")
    print(report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["smoke", "batch", "eval", "benchmark", "translate", "batch_nl"], default="batch")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--requests", default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--input-mode", choices=["auto", "fol", "nl"], default="auto")
    ap.add_argument("--no-z3", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.task == "smoke":
        run_unit_smoke(out)
        return

    loader_warnings = []
    if args.requests:
        records = list(records_from_batch(args.requests))
    elif args.dataset:
        data = load_json(args.dataset)
        input_mode = "nl" if args.task == "batch_nl" else args.input_mode
        records, loader_warnings = records_from_exact_dataset(data, input_mode=input_mode)
    else:
        raise SystemExit("Provide --dataset or --requests")

    if args.start:
        records = records[args.start:]
    if args.limit:
        records = records[: args.limit]

    llm = maybe_load_qwen()
    input_mode = "nl" if args.task == "batch_nl" else args.input_mode
    pipe = AnswerPipeline(llm=llm, input_mode=input_mode, use_z3=not args.no_z3)

    if args.task == "translate":
        pipe.input_mode = "nl"
        run_translation_benchmark(records, pipe, out)
    else:
        run_answer_batch(records, pipe, out, args.task, batch_size=args.batch_size, loader_warnings=loader_warnings)


if __name__ == "__main__":
    main()
