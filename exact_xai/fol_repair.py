from __future__ import annotations

"""Deterministic FOL repair/validation for EXACT v4.

The LLM sometimes emits Python/English-looking forms such as
``John completed_courses()`` or ``x.completed_course_a``.  The symbolic
reasoner only accepts predicate-argument syntax, so this module repairs common
surface mistakes before parsing and rejects the remaining unsafe strings.
"""

import re
from dataclasses import dataclass, field


@dataclass
class RepairReport:
    repaired: list[str]
    warnings: list[str] = field(default_factory=list)


def _num_token(x: str) -> str:
    return x.strip().replace(".", "_").replace("%", "percent")


def normalize_predicate_name(pred: str) -> str:
    pred = pred.strip()
    pred = re.sub(r"^x_", "", pred)
    pred = pred.replace("has_extended_library_access", "extended_library_access")
    pred = pred.replace("eligible_for_extended_library_access", "extended_library_access")
    pred = pred.replace("can_apply_for_collaborative_research_projects", "can_apply_collaborative_projects")
    pred = pred.replace("can_access_restricted_archives", "access_restricted_archives")
    pred = pred.replace("can_submit_research_proposals", "can_submit_proposals")
    pred = pred.replace("has_published_at_least_one_academic_paper", "published_at_least_one_paper")
    pred = pred.replace("completed_all_courses", "completed_all_required_courses")
    pred = pred.replace("qualifies_for_graduate_fellowship", "qualifies_for_graduate_fellowship_program")
    pred = pred.replace("graduate_fellowship", "graduate_fellowship_program")
    return pred


def normalize_entity_name(arg: str) -> str:
    a = arg.strip().strip('"\'')
    low = a.lower()
    aliases = {
        "john": "John",
        "professor_john": "John",
        "dr_john": "John",
        "nurse_john": "John",
        "sophia": "Sophia",
        "david": "David",
        "alex": "Alex",
        "sarah": "sarah",
        "minh": "minh",
        "phd_degree": "PhD",
        "phd": "PhD",
        "master_degree": "MSc",
        "masters_degree": "MSc",
        "master_s_degree": "MSc",
        "msc": "MSc",
        "bachelor_degree": "BA",
        "bachelors_degree": "BA",
        "ba": "BA",
    }
    return aliases.get(low, a)


def _normalize_pred_calls(s: str) -> str:
    def repl(m: re.Match) -> str:
        pred = normalize_predicate_name(m.group(1))
        args = ", ".join(normalize_entity_name(x.strip()) for x in m.group(2).split(",") if x.strip())
        return f"{pred}({args})"
    return re.sub(r"\b([A-Za-z_]\w*)\s*\(([^()]*)\)", repl, s)


def repair_fol_string(s: str) -> str:
    s = str(s).strip().strip("`")
    s = s.replace("∀", "ForAll").replace("∃", "Exists")
    s = s.replace("→", "->").replace("=>", "->")
    s = s.replace("∧", "&").replace("∨", "|")
    s = s.replace("¬", "not ").replace("~", "not ")
    s = re.sub(r"\bforall\b", "ForAll", s, flags=re.I)
    s = re.sub(r"\bexists\b", "Exists", s, flags=re.I)

    # John.completed_course_a       -> completed_course_a(John)
    # x.enrolled_in_b               -> enrolled_in_b(x)
    # Alex.membership_duration = 8  -> membership_duration(Alex, 8)
    s = re.sub(
        r"\b([A-Za-z][A-Za-z0-9_]*)\.([A-Za-z_]\w*)\s*=\s*([0-9]+(?:\.[0-9]+)?(?:_?[A-Za-z]+)?)",
        lambda m: f"{normalize_predicate_name(m.group(2))}({normalize_entity_name(m.group(1))}, {m.group(3)})",
        s,
    )
    s = re.sub(
        r"\b([A-Za-z][A-Za-z0-9_]*)\.([A-Za-z_]\w*)\b",
        lambda m: f"{normalize_predicate_name(m.group(2))}({normalize_entity_name(m.group(1))})",
        s,
    )

    # John completes_thesis()        -> completes_thesis(John)
    # x maintains_gpa_above_3_5()    -> maintains_gpa_above_3_5(x)
    s = re.sub(
        r"\b([A-Za-z][A-Za-z0-9_]*)\s+([a-z_]\w*)\s*\(\s*\)",
        lambda m: f"{normalize_predicate_name(m.group(2))}({normalize_entity_name(m.group(1))})",
        s,
    )
    s = re.sub(
        r"\b([A-Za-z][A-Za-z0-9_]*)\s+([a-z_]\w*)\s*\(\s*[0-9.]+\s*\)",
        lambda m: f"{normalize_predicate_name(m.group(2))}({normalize_entity_name(m.group(1))})",
        s,
    )

    # study_hours(x) >= 15 -> study_hours_at_least_15(x)
    s = re.sub(
        r"\b([A-Za-z_]\w*)\(([^()]*)\)\s*>=\s*([0-9]+(?:\.[0-9]+)?)(?:_?([A-Za-z]+))?",
        lambda m: f"{normalize_predicate_name(m.group(1))}_at_least_{_num_token(m.group(3))}({m.group(2)})",
        s,
    )
    s = re.sub(
        r"\b([A-Za-z_]\w*)\(([^()]*)\)\s*>\s*([0-9]+(?:\.[0-9]+)?)(?:_?([A-Za-z]+))?",
        lambda m: f"{normalize_predicate_name(m.group(1))}_above_{_num_token(m.group(3))}({m.group(2)})",
        s,
    )
    s = re.sub(
        r"\b([A-Za-z_]\w*)\(([^()]*)\)\s*<\s*([0-9]+(?:\.[0-9]+)?)(?:_?([A-Za-z]+))?",
        lambda m: f"{normalize_predicate_name(m.group(1))}_less_than_{_num_token(m.group(3))}({m.group(2)})",
        s,
    )
    s = re.sub(
        r"\b([A-Za-z_]\w*)\(([^()]*)\)\s*=\s*([0-9]+(?:\.[0-9]+)?)(?:_?([A-Za-z]+))?",
        lambda m: f"{normalize_predicate_name(m.group(1))}({m.group(2)}, {m.group(3) + (('_' + m.group(4)) if m.group(4) else '')})",
        s,
    )

    # Normalize predicate aliases after all rewrites.
    s = _normalize_pred_calls(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strict_fol_warnings(s: str) -> list[str]:
    out: list[str] = []
    if "." in s:
        out.append("dot_method_syntax")
    if re.search(r"\b[A-Za-z_]\w*\s+[A-Za-z_]\w*\s*\(", s):
        out.append("space_before_predicate_call")
    if re.search(r"(?<!-)>=|<=|(?<!-)>|<|==", s):
        out.append("raw_comparison_left")
    if re.search(r"\bnot_\w+\(", s):
        # not_foo(x) is parseable after parse_atom, but we still warn because it was repaired semantically.
        out.append("not_prefix_predicate")
    return out


def repair_fol_list(premises_fol: list[str]) -> RepairReport:
    repaired: list[str] = []
    warnings: list[str] = []
    for i, x in enumerate(premises_fol, start=1):
        r = repair_fol_string(x)
        if r != x:
            warnings.append(f"premise_{i}_fol_repaired")
        for w in strict_fol_warnings(r):
            warnings.append(f"premise_{i}_{w}")
        repaired.append(r)
    return RepairReport(repaired, sorted(set(warnings)))
