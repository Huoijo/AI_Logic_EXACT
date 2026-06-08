from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .fol import parse_fol_premises


@dataclass
class NL2LogicResult:
    premises_fol: list[str]
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _extract_json(text: str) -> Any | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def make_nl2logic_prompt(premises_nl: list[str], question: str | None = None) -> str:
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(premises_nl))
    return f"""
You are an NL-to-logic compiler for a proof-based QA system.
Convert every natural-language premise into a compact FOL-like string that this grammar accepts:
- Facts: predicate(Entity)
- Universal rules: ForAll(x, antecedent(x) -> consequent(x))
- Conjunction: A(x) & B(x)
- Negation: not A(x)
- Existential facts: Exists(x, predicate(x))

Rules:
- Use snake_case predicate names.
- Use the same predicate name for the same concept across all premises.
- Preserve premise order: output exactly one FOL string per input premise.
- Do not answer the question.
- Return ONLY valid JSON.

JSON schema:
{{"premises_fol": ["ForAll(x, ...)", "fact(Entity)"]}}

Natural-language premises:
{numbered}

Question context, only for vocabulary alignment:
{question or ""}
""".strip()


def _snake(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the|all|every|any|student|students|project|projects|python|code|is|are|has|have|been|then|if|who|that|it|its|for|to|of|and|or|with|required)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_") or "unknown"


def _heuristic_condition_to_atom(text: str, var: str = "x") -> str:
    t = text.strip().rstrip(".")
    neg = bool(re.search(r"\b(not|does not|do not|did not|without|no)\b", t, flags=re.I))
    pred = _snake(t)
    pred = re.sub(r"^(does_not|do_not|did_not|not)_", "", pred)
    return f"{'not ' if neg else ''}{pred}({var})"


def heuristic_translate_one(premise: str, premise_id: int) -> str:
    s = premise.strip().rstrip(".")

    # If A, then B.
    m = re.match(r"if\s+(.+?),\s*then\s+(.+)$", s, flags=re.I)
    if m:
        left = _heuristic_condition_to_atom(m.group(1), "x")
        right = _heuristic_condition_to_atom(m.group(2), "x")
        return f"ForAll(x, {left} -> {right})"

    # X who/that A are B.  Example: Students who completed X are eligible.
    m = re.match(r"(?:students|people|applicants|projects|python projects|python code)\s+who\s+(.+?)\s+(?:are|is)\s+(.+)$", s, flags=re.I)
    if m:
        left = _heuristic_condition_to_atom(m.group(1), "x")
        right = _heuristic_condition_to_atom(m.group(2), "x")
        return f"ForAll(x, {left} -> {right})"

    # All X are Y / Every X is Y.
    m = re.match(r"(?:all|every)\s+(.+?)\s+(?:are|is)\s+(.+)$", s, flags=re.I)
    if m:
        pred = _heuristic_condition_to_atom(m.group(2), "x")
        return f"ForAll(x, {pred})"

    # There exists at least one X that/is/has Y.
    m = re.match(r"there exists(?: at least)? one .+?(?:that|who|which)?\s*(?:is|are|has|have|follows|follow)?\s*(.+)$", s, flags=re.I)
    if m:
        pred = _heuristic_condition_to_atom(m.group(1), "x")
        return f"Exists(x, {pred})"

    # Named fact: Sophia has completed her capstone project.
    ent = None
    m = re.match(r"([A-Z][A-Za-z0-9_]*)\s+(.+)$", s)
    if m:
        ent = m.group(1)
        pred = _snake(m.group(2))
        return f"{pred}({ent})"

    return f"unknown_premise_{premise_id}(GENERIC)"


def heuristic_translate_premises(premises_nl: list[str]) -> NL2LogicResult:
    fol = [heuristic_translate_one(p, i + 1) for i, p in enumerate(premises_nl)]
    return NL2LogicResult(fol, warnings=["used_heuristic_nl2logic"])


def validate_fol_list(premises_fol: list[str]) -> tuple[bool, list[str]]:
    warnings = []
    if not isinstance(premises_fol, list) or not all(isinstance(x, str) for x in premises_fol):
        return False, ["premises_fol_not_list_of_strings"]
    try:
        parse_fol_premises(premises_fol)
    except Exception as e:
        return False, [f"fol_parse_error: {type(e).__name__}: {e}"]
    return True, warnings


def translate_nl_to_fol(premises_nl: list[str], question: str | None = None, llm=None) -> NL2LogicResult:
    if not premises_nl:
        return NL2LogicResult([], warnings=["no_nl_premises"])

    if llm is not None:
        prompt = make_nl2logic_prompt(premises_nl, question)
        try:
            text = llm.generate(prompt, max_new_tokens=1024, temperature=0.0)
            obj = _extract_json(text)
            if isinstance(obj, dict):
                fol = obj.get("premises_fol") or obj.get("premises-FOL") or []
                ok, warns = validate_fol_list(fol)
                if ok and len(fol) == len(premises_nl):
                    return NL2LogicResult(fol, warnings=warns, raw={"llm_text": text})
                if ok:
                    return NL2LogicResult(fol, warnings=warns + ["nl2logic_count_mismatch"], raw={"llm_text": text})
                # one repair attempt: ask it to fix only format/count.
                repair_prompt = prompt + "\n\nYour previous output was invalid. Return valid JSON only with exactly one FOL string per premise. Previous output:\n" + text
                text2 = llm.generate(repair_prompt, max_new_tokens=1024, temperature=0.0)
                obj2 = _extract_json(text2)
                if isinstance(obj2, dict):
                    fol2 = obj2.get("premises_fol") or obj2.get("premises-FOL") or []
                    ok2, warns2 = validate_fol_list(fol2)
                    if ok2:
                        return NL2LogicResult(fol2, warnings=warns2 + ["nl2logic_repaired"], raw={"llm_text": text, "repair_text": text2})
        except Exception as e:
            return heuristic_translate_premises(premises_nl)

    return heuristic_translate_premises(premises_nl)
