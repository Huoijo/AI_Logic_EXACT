from __future__ import annotations

import json
import re
from typing import Any

from rapidfuzz import fuzz

from .fol import Atom, KnowledgeBase, parse_atom
from .fol_repair import repair_fol_string, normalize_predicate_name
from .schemas import ParsedQuestion

# Match EXACT-style MCQ options of the form A. ... B. ... C. ... D. ...
CHOICE_RE = re.compile(r"(?m)^\s*([A-D])\.\s*(.+?)(?=\n\s*[A-D]\.\s*|\Z)", re.S)
YES_NO_STARTERS = (
    "does ", "do ", "is ", "are ", "can ", "could ",
    "should ", "would ", "will ", "did ", "has ", "have ",
)
NEGATION_PATTERNS = (
    r"\bnot\b",
    r"\bno\b",
    r"\bnever\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bdoes\s+not\b",
    r"\bdo\s+not\b",
    r"\bdid\s+not\b",
    r"\bis\s+not\b",
    r"\bare\s+not\b",
    r"\bhas\s+not\b",
    r"\bhave\s+not\b",
    r"\bneeds?\b",  # options like "needs recommendation" usually query the requirement fact
)

DOMAIN_REWRITES = {
    "pep 8": "pep8",
    "pep-8": "pep8",
    "well tested": "well_tested",
    "well-tested": "well_tested",
    "well structured": "well_structured",
    "well-structured": "well_structured",
    "clean and readable": "clean_readable",
    "clean readable": "clean_readable",
    "clean code": "clean_code",
    "easy to maintain": "easy_to_maintain",
    "international program": "international_program",
    "university scholarship": "university_scholarship",
    "scholarship": "scholarship",
    "honors diploma": "honors_diploma",
    "advanced courses": "advanced_courses",
    "faculty recommendation": "faculty_recommendation",
    "language proficiency": "language_proficiency",
    "capstone project": "capstone_project",
    "community service": "community_service",
    "core curriculum": "core_curriculum",
    "science assessment": "science_assessment",
    "research methodology": "research_methodology",
    "graduate fellowship": "graduate_fellowship",
    "academic distinction": "academic_distinction",
    "graduate courses": "graduate_courses",
    "undergraduate courses": "undergraduate_courses",
    "research mentor": "research_mentor",
    "curriculum committees": "curriculum_committees",
    "new courses": "new_courses",
    "restricted archives": "restricted_archives",
    "research proposals": "research_proposals",
    "collaborative research projects": "collaborative_research_projects",
    "hazardous materials": "hazardous_materials",
    "hazardous cargo": "hazardous_cargo",
    "state lines": "state_lines",
    "standard goods": "standard_goods",
    "safety endorsement": "safety_endorsement",
    "research fellowship program": "graduate_fellowship_program",
    "graduate fellowship program": "graduate_fellowship_program",
    "academic papers": "publications",
    "personal training sessions": "book_training",
    "training sessions": "book_training",
    "advanced classes": "advanced_classes",
    "clinical hours": "clinical_hours",
}


def normalize_text_for_matching(s: str) -> str:
    s = str(s).lower()
    for a, b in DOMAIN_REWRITES.items():
        s = s.replace(a, b)
    return s


def _snake(s: str) -> str:
    s = normalize_text_for_matching(s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def normalize_logic_value(s: str) -> str:
    """Normalize model-produced logic strings into parser/Z3-friendly syntax."""
    return repair_fol_string(str(s).strip())


def extract_choices(question: str) -> dict[str, str]:
    return {m.group(1): " ".join(m.group(2).split()) for m in CHOICE_RE.finditer(question)}


def has_mcq_options(question: str) -> bool:
    choices = extract_choices(question)
    return "A" in choices and "B" in choices


def looks_yes_no(question: str) -> bool:
    return question.strip().lower().startswith(YES_NO_STARTERS)


def predicates(kb: KnowledgeBase) -> set[str]:
    ps = {f.pred for f in kb.facts}
    for r in kb.rules:
        ps.add(r.consequent.pred)
        ps.update(a.pred for a in r.antecedents)
    return {p for p in ps if p}


def constants(kb: KnowledgeBase) -> set[str]:
    return kb.constants()


def _real_constants(kb: KnowledgeBase) -> list[str]:
    return sorted(c for c in constants(kb) if not c.startswith("EXISTS_") and c != "GENERIC")


def _default_constant(kb: KnowledgeBase) -> str:
    real = _real_constants(kb)
    if real:
        return real[0]
    cs = sorted(constants(kb))
    return cs[0] if cs else "GENERIC"


def best_predicate_from_text(text: str, kb: KnowledgeBase, threshold: int = 55) -> str | None:
    text_norm = _snake(text)
    if not text_norm:
        return None

    best = None
    best_score = 0.0
    for p in predicates(kb):
        p_norm = _snake(p)
        score = max(
            fuzz.partial_ratio(p_norm, text_norm),
            fuzz.partial_ratio(text_norm, p_norm),
            fuzz.token_set_ratio(p_norm.replace("_", " "), normalize_text_for_matching(text)),
        )
        if score > best_score:
            best = p
            best_score = score
    return best if best_score >= threshold else None


def best_constant_from_text(text: str, kb: KnowledgeBase) -> str | None:
    best = None
    best_score = 0.0
    text_l = text.lower()
    for c in _real_constants(kb):
        score = fuzz.partial_ratio(c.lower(), text_l)
        # Professor John / Dr. John should map to John if John is the KB constant.
        if c.lower() in text_l:
            score = max(score, 100)
        if score > best_score:
            best = c
            best_score = score
    return best if best_score >= 75 else None


def _is_negated(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in NEGATION_PATTERNS)


def _remove_negation_words(text: str) -> str:
    t = text
    t = re.sub(r"\bdoes\s+not\b|\bdo\s+not\b|\bdid\s+not\b", "", t, flags=re.I)
    t = re.sub(r"\bis\s+not\b|\bare\s+not\b|\bhas\s+not\b|\bhave\s+not\b", "", t, flags=re.I)
    t = re.sub(r"\bcannot\b|\bcan't\b|\bnot\b|\bno\b|\bnever\b", "", t, flags=re.I)
    # For "needs X" we usually want predicate X(Entity), not not X(Entity).
    t = re.sub(r"\bneeds?\b|\bmust\b|\bto qualify\b|\bto get\b", "", t, flags=re.I)
    return " ".join(t.split())


def _clean_phrase(text: str) -> str:
    t = text.strip()
    t = re.sub(
        r"^(all|a|an|the|any|someone|anyone|students?|faculty members?|drivers?|python projects?|python code)\b",
        "",
        t,
        flags=re.I,
    )
    t = re.sub(r"\b(according to the premises|according to the above premises)\b", "", t, flags=re.I)
    t = re.sub(
        r"\b(must|can|could|should|would|will|does|do|is|are|has|have|then|that|it|they|he|she)\b",
        "",
        t,
        flags=re.I,
    )
    return " ".join(t.split())


def _phrase_to_atom_formula(text: str, kb: KnowledgeBase, arg: str = "x") -> str | None:
    neg = _is_negated(text)
    needs_more = bool(re.search(r"\bneeds?\s+(more|additional|longer|extra)\b", text, flags=re.I))
    needs_plain = bool(re.search(r"\bneeds?\b", text, flags=re.I)) and not needs_more
    cleaned = _clean_phrase(_remove_negation_words(text) if (neg or needs_plain or needs_more) else text)
    pred = best_predicate_from_text(cleaned, kb) or best_predicate_from_text(text, kb)
    if not pred:
        return None
    pred = normalize_predicate_name(pred)
    # "needs more/additional X" means the positive requirement is not yet satisfied.
    if needs_more:
        return f"not {pred}({arg})"
    # "needs X" often asks whether X is the missing requirement, so query X itself.
    if neg and not needs_plain:
        return f"not {pred}({arg})"
    return f"{pred}({arg})"


def _phrase_to_atom(text: str, kb: KnowledgeBase, default_const: str | None = None) -> str | None:
    const = best_constant_from_text(text, kb) or default_const or _default_constant(kb)
    return _phrase_to_atom_formula(text, kb, const)


def _split_if_then(text: str) -> tuple[str, str] | None:
    t = " ".join(text.strip().split())
    m = re.search(r"\bif\b\s+(.+?)\s*,?\s*\bthen\b\s+(.+)$", t, flags=re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip().rstrip(".?")
    return None


def _is_explicit_conditional(text: str) -> bool:
    return _split_if_then(text) is not None


def _parse_conditional_as_forall(text: str, kb: KnowledgeBase) -> str | None:
    parts = _split_if_then(text)
    if not parts:
        return None
    left, right = parts
    ant = _phrase_to_atom_formula(left, kb, "x")
    cons = _phrase_to_atom_formula(right, kb, "x")
    if ant and cons:
        return f"ForAll(x, {ant} -> {cons})"
    return None


def _extract_embedded_conditional(question: str) -> str | None:
    q = question.strip()
    m = re.search(r"\bthat\b\s+(.+?)\??$", q, flags=re.I | re.S)
    if m and " if " in f" {m.group(1).lower()} ":
        return m.group(1).strip().rstrip("?")
    if q.lower().startswith("if "):
        return q.rstrip("?")
    return None


def parse_question_rule_based(question: str, kb: KnowledgeBase) -> ParsedQuestion:
    choices = extract_choices(question)
    if choices:
        parsed_choices: dict[str, str] = {}
        context_const = best_constant_from_text(question, kb)
        for label, text in choices.items():
            conditional = _parse_conditional_as_forall(text, kb)
            if conditional:
                parsed_choices[label] = conditional
                continue
            atom = _phrase_to_atom(text, kb, context_const)
            parsed_choices[label] = atom if atom else _snake(text)
        return ParsedQuestion(kind="multiple_choice", choices=parsed_choices, parser="rule_based")

    embedded = _extract_embedded_conditional(question)
    if embedded:
        conditional = _parse_conditional_as_forall(embedded, kb)
        if conditional:
            return ParsedQuestion(kind="yes_no", target=conditional, parser="rule_based")

    target = _phrase_to_atom(question, kb)
    return ParsedQuestion(kind="yes_no", target=target, parser="rule_based")


def make_llm_prompt(question: str, kb: KnowledgeBase) -> str:
    preds = sorted(predicates(kb))
    consts = sorted(constants(kb))
    choices = extract_choices(question)
    expected_kind = "multiple_choice" if choices else "yes_no"
    choices_block = "\n".join(f"{k}. {v}" for k, v in choices.items()) if choices else "<none>"

    return f"""
/no_think
You are a strict NL-to-FOL query compiler for a symbolic reasoner.
Return ONLY valid JSON. Do not solve the question. Do not choose an answer.

Available predicates:
{preds}

Known constants/entities:
{consts}

Original question:
{question}

Detected MCQ choices:
{choices_block}

Hard constraints:
- The expected kind is "{expected_kind}".
- If there are no A/B/C/D choices in the original question, you MUST output kind="yes_no".
- Never invent A/B choices for a Yes/No question.
- If the original question has A-D choices, output kind="multiple_choice" and include exactly A, B, C, D.
- Parse each MCQ option independently.
- Use ForAll implication ONLY when the option itself is an explicit "If ..., then ..." statement.
- For factual statements about named entities, output an atom like predicate(Entity), never ForAll(...).
- If an option says "If ..., then ...", the output for that option MUST contain "->".
- Use only available predicates when possible.
- Prefer canonical syntax: ForAll(x, A(x) -> B(x)); use "not pred(x)" for negation.
- Do not map all options to the same target unless the option texts are truly equivalent.

Schema for Yes/No:
{{"kind":"yes_no","target":"predicate(Entity) or ForAll(x, A(x) -> B(x))"}}

Schema for multiple choice:
{{"kind":"multiple_choice","choices":{{"A":"...","B":"...","C":"...","D":"..."}}}}
""".strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = re.sub(r"```(?:json)?", "", text).replace("```", "")
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _normalize_parsed_obj(obj: dict[str, Any]) -> dict[str, Any] | None:
    kind = obj.get("kind")
    if kind not in {"yes_no", "multiple_choice", "open"}:
        return None

    if kind == "multiple_choice":
        choices = obj.get("choices") or {}
        if not isinstance(choices, dict):
            return None
        labels = set(choices.keys())
        if not {"A", "B", "C", "D"}.issubset(labels):
            return None
        obj["choices"] = {k: normalize_logic_value(v) for k, v in choices.items() if k in {"A", "B", "C", "D"}}
        obj["target"] = None
        return obj

    if kind == "yes_no":
        target = obj.get("target")
        if not target or not isinstance(target, str):
            return None
        obj["target"] = normalize_logic_value(target)
        obj["choices"] = {}
        return obj

    return obj


def parse_llm_json(text: str) -> ParsedQuestion | None:
    obj = _extract_json_object(text)
    if obj is None:
        return None
    obj = _normalize_parsed_obj(obj)
    if obj is None:
        return None
    try:
        return ParsedQuestion(**obj, raw={"llm_text": text}, parser="llm")
    except Exception:
        return None


def postprocess_parsed_question(question: str, kb: KnowledgeBase, parsed: ParsedQuestion) -> ParsedQuestion:
    """Enforce shape constraints after LLM parsing.

    This is intentionally deterministic. It fixes the common failure where the
    model converts factual options like "Sophia qualifies..." into universal
    implications, which makes material-implication reasoning too permissive.
    """
    post_warnings: list[str] = []
    choices_text = extract_choices(question)

    # Yes/No guard: no A/B/C/D in the source question means the output must be yes_no.
    if not choices_text:
        if parsed.kind != "yes_no":
            post_warnings.append("forced_yes_no_over_llm_kind")
            rb = parse_question_rule_based(question, kb)
            rb.raw = {**parsed.raw, "postprocess_warnings": post_warnings}
            rb.parser = f"{parsed.parser}+guard"
            return rb
        embedded = _extract_embedded_conditional(question)
        if embedded:
            conditional = _parse_conditional_as_forall(embedded, kb)
            if conditional and parsed.target != conditional:
                parsed.target = conditional
                post_warnings.append("normalized_yes_no_conditional_target")

        # If the source is not an explicit conditional/statement question, reject hallucinated ForAll targets.
        if parsed.target and ("->" in parsed.target or parsed.target.lower().startswith("forall")):
            qlow = question.lower()
            has_statement = "statement:" in qlow or _extract_embedded_conditional(question) is not None
            if not has_statement and any(k in qlow for k in ["meet", "meets", "can ", "qualify", "eligible", "demonstrate"]):
                atom = _phrase_to_atom(question, kb)
                if atom:
                    parsed.target = atom
                    post_warnings.append("normalized_yes_no_forall_to_atom")
        parsed.raw = {**parsed.raw, "postprocess_warnings": post_warnings}
        return parsed

    # MCQ guard: source has choices, so output must be multiple_choice A-D.
    if parsed.kind != "multiple_choice" or not parsed.choices:
        rb = parse_question_rule_based(question, kb)
        rb.raw = {**parsed.raw, "postprocess_warnings": ["forced_mcq_over_llm_kind"]}
        rb.parser = f"{parsed.parser}+guard"
        return rb

    fixed: dict[str, str] = {}
    context_const = best_constant_from_text(question, kb)
    for label, option_text in choices_text.items():
        model_value = normalize_logic_value(parsed.choices.get(label, ""))
        option_is_conditional = _is_explicit_conditional(option_text)
        option_const = best_constant_from_text(option_text, kb) or context_const

        if option_is_conditional:
            # Explicit conditionals should be ForAll implications. If the model failed,
            # reconstruct from the option text.
            if "->" in model_value and (model_value.lower().startswith("forall") or "(" in model_value):
                fixed[label] = model_value
            else:
                fallback = _parse_conditional_as_forall(option_text, kb)
                fixed[label] = fallback if fallback else model_value
                post_warnings.append(f"option_{label}_conditional_reparsed")
            continue

        # Factual entity options should be atoms. ForAll here is usually a hallucinated rule.
        if option_const or model_value.lower().startswith("forall") or "->" in model_value:
            atom = _phrase_to_atom(option_text, kb, context_const)
            if atom:
                fixed[label] = atom
                if model_value and model_value != atom:
                    post_warnings.append(f"option_{label}_entity_atom_postprocessed")
                continue

        # Last fallback: keep model value if it is atom-like; otherwise use rule-based atom.
        if model_value and "->" not in model_value:
            fixed[label] = model_value
        else:
            atom = _phrase_to_atom(option_text, kb, context_const)
            fixed[label] = atom if atom else (model_value or _snake(option_text))
            post_warnings.append(f"option_{label}_fallback_atom")

    parsed.choices = fixed
    parsed.target = None
    parsed.raw = {**parsed.raw, "postprocess_warnings": post_warnings}
    if post_warnings:
        parsed.parser = f"{parsed.parser}+post"
    return parsed


def parsed_target_to_atom(s: str | None) -> Atom | None:
    if not s:
        return None
    q = s.strip()
    if "->" in q or "→" in q or q.lower().startswith("forall") or q.startswith("∀"):
        return None
    try:
        return parse_atom(q)
    except Exception:
        return None
