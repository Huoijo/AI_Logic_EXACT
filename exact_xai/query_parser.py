from __future__ import annotations

import json
import re
from rapidfuzz import fuzz
from .fol import Atom, KnowledgeBase, parse_atom
from .schemas import ParsedQuestion

CHOICE_RE = re.compile(r"(?m)^\s*([A-D])\.\s*(.+?)(?=\n\s*[A-D]\.\s*|\Z)", re.S)

def _snake(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")

def extract_choices(question: str) -> dict[str, str]:
    return {m.group(1): " ".join(m.group(2).split()) for m in CHOICE_RE.finditer(question)}

def predicates(kb: KnowledgeBase) -> set[str]:
    ps = {f.pred for f in kb.facts}
    for r in kb.rules:
        ps.add(r.consequent.pred)
        ps.update(a.pred for a in r.antecedents)
    return ps

def constants(kb: KnowledgeBase) -> set[str]:
    return kb.constants()

def best_predicate_from_text(text: str, kb: KnowledgeBase) -> str | None:
    text_norm = _snake(text)
    best = None
    best_score = 0
    for p in predicates(kb):
        score = max(fuzz.partial_ratio(_snake(p), text_norm), fuzz.partial_ratio(text_norm, _snake(p)))
        score = max(score, fuzz.token_set_ratio(_snake(p).replace("_", " "), text.lower()))
        if score > best_score:
            best = p; best_score = score
    return best if best_score >= 55 else None

def best_constant_from_text(text: str, kb: KnowledgeBase) -> str | None:
    best = None; best_score = 0
    for c in constants(kb):
        if c.startswith("EXISTS_") or c == "GENERIC":
            continue
        score = fuzz.partial_ratio(c.lower(), text.lower())
        if score > best_score:
            best = c; best_score = score
    return best if best_score >= 75 else None

def parse_question_rule_based(question: str, kb: KnowledgeBase) -> ParsedQuestion:
    choices = extract_choices(question)
    if choices:
        parsed_choices = {}
        for label, text in choices.items():
            pred = best_predicate_from_text(text, kb)
            const = best_constant_from_text(text, kb) or next(iter(constants(kb)))
            if pred:
                parsed_choices[label] = f"{pred}({const})"
            else:
                parsed_choices[label] = _snake(text)
        return ParsedQuestion(kind="multiple_choice", choices=parsed_choices, parser="rule_based")

    pred = best_predicate_from_text(question, kb)
    const = best_constant_from_text(question, kb) or next(iter(constants(kb)))
    target = f"{pred}({const})" if pred else None
    return ParsedQuestion(kind="yes_no", target=target, parser="rule_based")

def make_llm_prompt(question: str, kb: KnowledgeBase) -> str:
    preds = sorted(predicates(kb))
    consts = sorted(constants(kb))
    return f"""
You convert a question into formal query JSON for a symbolic reasoner.
Return ONLY valid JSON.

Available predicates: {preds}
Known constants/entities: {consts}

Question:
{question}

Output rules:
- For Yes/No questions, output a direct target atom when possible.
- For multiple-choice questions, parse EACH option independently.
- A choice value may be either an atom like predicate(Entity) or a universal implication like ForAll(x, A(x) -> B(x)).
- Do not map all options to the same target unless the option texts are truly equivalent.

Schema for Yes/No:
{{"kind":"yes_no","target":"predicate(Entity)"}}

Schema for multiple choice:
{{"kind":"multiple_choice","choices":{{"A":"predicate(Entity)","B":"ForAll(x, A(x) -> B(x))"}}}}
""".strip()

def parse_llm_json(text: str) -> ParsedQuestion | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return ParsedQuestion(**obj, raw={"llm_text": text}, parser="llm")
    except Exception:
        return None

def parsed_target_to_atom(s: str | None) -> Atom | None:
    if not s:
        return None
    return parse_atom(s)
