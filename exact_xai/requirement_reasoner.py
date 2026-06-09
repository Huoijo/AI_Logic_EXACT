from __future__ import annotations

"""Extra conservative requirement reasoning for EXACT v4.

Forward chaining answers Yes when all requirements are derivable.  This module
helps answer No for questions phrased as "meets all requirements" when a direct
required antecedent is explicitly negated or simply not derivable.
"""

from .fol import Atom, KnowledgeBase, Rule, parse_atom, is_var
from .reasoner import Reasoner, ReasonResult
from .schemas import ProofStep


def _unify(pattern: Atom, target: Atom) -> dict[str, str] | None:
    if pattern.pred != target.pred or pattern.negated != target.negated or len(pattern.args) != len(target.args):
        return None
    env: dict[str, str] = {}
    for pa, ta in zip(pattern.args, target.args):
        if is_var(pa):
            env[pa] = ta
        elif pa != ta:
            return None
    return env


def question_requests_requirements(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in [
        "meet all requirements",
        "meets all requirements",
        "all requirements",
        "current eligibility status",
    ])


def requirement_gap_check(kb: KnowledgeBase, target: Atom, reasoner: Reasoner) -> ReasonResult | None:
    candidates: list[tuple[Rule, dict[str, str]]] = []
    for rule in kb.rules:
        env = _unify(rule.consequent, target)
        if env is not None and rule.antecedents:
            candidates.append((rule, env))
    if not candidates:
        return None

    # If every direct derivation route has at least one missing/blocked required condition,
    # then the entity does not meet all requirements for target.
    missing_notes: list[str] = []
    used: set[int] = set()
    for rule, env in candidates:
        route_missing = []
        route_used = {rule.source_id}
        for ant in rule.antecedents:
            grounded = ant.substitute(env)
            rr = reasoner.prove_atom(grounded)
            if rr.answer != "Yes":
                route_missing.append(str(grounded))
            else:
                route_used.update(rr.used_premises)
        if not route_missing:
            return None
        missing_notes.extend(route_missing)
        used.update(route_used)

    note = "Missing or blocked required condition(s): " + ", ".join(sorted(set(missing_notes)))
    proof = [ProofStep(derived=f"not {target}", rule_id=None, used=sorted(set(missing_notes)), used_premises=sorted(used), note=note)]
    return ReasonResult("No", proof=proof, used_premises=sorted(used), warnings=["requirement_gap_no"])
