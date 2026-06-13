from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import re

from .schemas import AnswerRequest

MCQ_ANSWER_RE = re.compile(r"^[A-D]$", re.I)
# EXACT datasets use both "Unknown" and "Uncertain" for the same
# third truth value.  Keep them canonicalized for metrics/alignment.
UNKNOWN_SET = {"unknown", "uncertain"}
YESNO_SET = {"yes", "no"}
# Support common MCQ spellings: A. ..., A) ..., A: ...
CHOICE_LINE_RE = re.compile(r"(?m)^\s*([A-D])\s*[\.\):]\s+.+$")


def infer_question_kind(question: str) -> str:
    q = question or ""
    labels = {m.group(1).upper() for m in CHOICE_LINE_RE.finditer(q)}
    if {"A", "B"}.issubset(labels):
        return "multiple_choice"
    return "yes_no"


def canonical_answer(value: Any) -> str | None:
    """Canonicalize labels used by reports/metrics.

    The benchmark data mixes "Unknown" and "Uncertain".  The solver emits
    "Uncertain", while some gold labels use "Unknown"; they are equivalent.
    """
    if value is None:
        return None
    s = str(value).strip()
    low = s.lower()
    if low in UNKNOWN_SET:
        return "uncertain"
    if low in YESNO_SET:
        return low
    if MCQ_ANSWER_RE.match(s):
        return s.upper()
    return low


def answers_equivalent(pred: Any, gold: Any) -> bool:
    if gold is None:
        return False
    return canonical_answer(pred) == canonical_answer(gold)


def infer_answer_kind(answer: Any) -> str:
    if answer is None:
        return "unknown"
    s = str(answer).strip()
    low = s.lower()
    if low in YESNO_SET:
        return "yes_no"
    if low in UNKNOWN_SET:
        # Unknown/Uncertain can be the correct non-provable answer for either
        # Yes/No or MCQ records, so alignment treats it as compatible with the
        # same-position question instead of as an "open" answer.
        return "unknown"
    if MCQ_ANSWER_RE.match(s):
        return "multiple_choice"
    return "open"


def _answer_candidates_from_record(rec: dict[str, Any]) -> list[Any]:
    """Return answers regardless of whether the dataset stores them as a list or split keys.

    EXACT-like records are sometimes inconsistent: answers may be in the same order as
    questions, or the MCQ and Yes/No answers may be interchanged. Some generated
    variants also use keys like `MCQ_answer` and `Yes/No_answer`. This helper keeps the
    loader tolerant and lets `align_answers_to_questions` repair the order.
    """
    if isinstance(rec.get("answers"), list):
        return list(rec["answers"])

    out: list[Any] = []
    split_keys = [
        "MCQ_answer", "mcq_answer", "multiple_choice_answer",
        "Yes/No_answer", "yes_no_answer", "yesno_answer", "YN_answer",
        "answer", "gold_answer",
    ]
    for k in split_keys:
        if k in rec and rec[k] is not None:
            out.append(rec[k])
    return out


def align_answers_to_questions(questions: list[str], answers: list[Any]) -> tuple[list[Any | None], list[str]]:
    """Align gold answers to questions by type.

    If question 0 is MCQ but answer[0] is Yes/No, and question 1 is Yes/No but
    answer[1] is A/B/C/D, this function swaps them. If everything already matches,
    it preserves the original order.
    """
    warnings: list[str] = []
    used: set[int] = set()
    aligned: list[Any | None] = []

    for i, q in enumerate(questions):
        qkind = infer_question_kind(q)

        # Prefer same position if type matches.  Unknown/Uncertain is allowed
        # for either MCQ or Yes/No because generated EXACT records use it as
        # the third truth value / no-provable-option marker.
        if i < len(answers) and i not in used:
            same_kind = infer_answer_kind(answers[i])
            if same_kind == qkind or same_kind == "unknown":
                aligned.append(answers[i])
                used.add(i)
                continue

        # Search another unused answer with the matching type.
        found = None
        for j, ans in enumerate(answers):
            if j in used:
                continue
            if infer_answer_kind(ans) == qkind:
                found = j
                break
        if found is not None:
            aligned.append(answers[found])
            used.add(found)
            warnings.append(f"answer_alignment: question {i} expected {qkind}, used answer index {found}")
            continue

        # Fallback: keep same position or None.
        if i < len(answers) and i not in used:
            aligned.append(answers[i])
            used.add(i)
            warnings.append(f"answer_alignment_fallback: question {i} expected {qkind}, same-position answer type={infer_answer_kind(answers[i])}")
        else:
            aligned.append(None)
            warnings.append(f"answer_alignment_missing: question {i} expected {qkind}")

    return aligned, warnings


def records_from_exact_dataset(data: list[dict[str, Any]], input_mode: str = "auto") -> tuple[list[AnswerRequest], list[str]]:
    """Convert EXACT dataset records into AnswerRequest rows.

    input_mode:
      - fol: always use premises-FOL if present
      - nl:  ignore premises-FOL and force NL->logic translation
      - auto: use FOL when present, otherwise NL
    """
    all_warnings: list[str] = []
    requests: list[AnswerRequest] = []
    for rid, rec in enumerate(data):
        questions = list(rec.get("questions", []))
        answers = _answer_candidates_from_record(rec)
        aligned, warns = align_answers_to_questions(questions, answers)
        all_warnings.extend([f"record {rid}: {w}" for w in warns])

        premises_nl = rec.get("premises-NL", []) or rec.get("premises_nl", []) or []
        premises_fol = rec.get("premises-FOL", []) or rec.get("premises_fol", []) or []
        if input_mode == "nl":
            premises_fol = []
        elif input_mode == "auto":
            premises_fol = premises_fol or []
        elif input_mode == "fol":
            pass
        else:
            raise ValueError(f"Unknown input_mode={input_mode}")

        for qid, q in enumerate(questions):
            requests.append(AnswerRequest(
                id=f"{rid}:{qid}",
                premises_nl=premises_nl,
                premises_fol=premises_fol,
                question=q,
                question_type=infer_question_kind(q),
                gold_answer=aligned[qid] if qid < len(aligned) else None,
            ))
    return requests, all_warnings
