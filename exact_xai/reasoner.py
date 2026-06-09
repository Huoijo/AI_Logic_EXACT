from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from .fol import Atom, KnowledgeBase, parse_atom, is_var
from .schemas import ProofStep

@dataclass
class ReasonResult:
    answer: str
    proof: list[ProofStep] = field(default_factory=list)
    used_premises: list[int] = field(default_factory=list)
    derived_facts: set[Atom] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)

def _num_from_token(x: str) -> float | None:
    import re
    m = re.search(r"(\d+(?:_\d+|\.\d+)?)", str(x))
    if not m:
        return None
    try:
        return float(m.group(1).replace("_", "."))
    except Exception:
        return None


def _numeric_pred_parts(pred: str):
    """Return skeleton and threshold for predicates like completing_600_hours."""
    import re
    m = re.match(r"^(.*?)(\d+(?:_\d+)?)(.*)$", pred)
    if not m:
        return None
    prefix, num, suffix = m.groups()
    try:
        val = float(num.replace("_", "."))
    except Exception:
        return None
    return prefix, suffix, val


def _pred_entails(fact_pred: str, pattern_pred: str) -> bool:
    if fact_pred == pattern_pred:
        return True
    fp = _numeric_pred_parts(fact_pred)
    pp = _numeric_pred_parts(pattern_pred)
    if fp and pp and fp[0] == pp[0] and fp[1] == pp[1]:
        return fp[2] >= pp[2]
    # completing_600_clinical_hours entails completing_500_clinical_hours.
    return False


def _args_match(pattern_args, fact_args, env: dict[str, str]) -> bool:
    if len(pattern_args) != len(fact_args):
        return False
    for pa, fa in zip(pattern_args, fact_args):
        # Numeric threshold in second argument: completed_courses(x,5) matches completed_courses(Sarah,4) only if 4 >= 5.
        if is_var(pa):
            if pa in env and env[pa] != fa:
                return False
            env[pa] = fa
        elif pa != fa:
            pn = _num_from_token(pa)
            fn = _num_from_token(fa)
            if pn is not None and fn is not None:
                if fn < pn:
                    return False
            else:
                return False
    return True


def _match_fact(pattern: Atom, facts: set[Atom]) -> list[dict[str, str]]:
    envs = []
    for f in facts:
        if f.negated != pattern.negated or len(f.args) != len(pattern.args):
            continue
        if not _pred_entails(f.pred, pattern.pred):
            continue
        env: dict[str, str] = {}
        if _args_match(pattern.args, f.args, env):
            envs.append(env)
    return envs

def _merge_envs(a: dict[str, str], b: dict[str, str]) -> dict[str, str] | None:
    out = dict(a)
    for k, v in b.items():
        if k in out and out[k] != v:
            return None
        out[k] = v
    return out

class Reasoner:
    def __init__(self, kb: KnowledgeBase, max_steps: int = 256):
        self.kb = kb
        self.max_steps = max_steps

    def closure(self) -> tuple[set[Atom], dict[Atom, ProofStep]]:
        facts = set(self.kb.facts)
        proofs: dict[Atom, ProofStep] = {
            f: ProofStep(
                derived=str(f),
                rule_id=None,
                used=[],
                used_premises=sorted(set(self.kb.fact_sources.get(f, []))),
                note=(f"Given by premise(s) {self.kb.fact_sources.get(f, [])}" if self.kb.fact_sources.get(f) else "Given fact"),
            )
            for f in facts
        }
        constants = self.kb.constants()
        changed = True
        steps = 0
        while changed and steps < self.max_steps:
            changed = False
            steps += 1
            for rule in self.kb.rules:
                envs = [dict()]
                if not rule.antecedents:
                    vars_ = sorted(rule.consequent.variables)
                    combos = product(constants, repeat=len(vars_)) if vars_ else [()]
                    envs = [dict(zip(vars_, combo)) for combo in combos]
                else:
                    for ant in rule.antecedents:
                        matches = _match_fact(ant, facts)
                        new_envs = []
                        for e in envs:
                            for m in matches:
                                merged = _merge_envs(e, m)
                                if merged is not None:
                                    new_envs.append(merged)
                        envs = new_envs
                        if not envs:
                            break
                for env in envs:
                    derived = rule.consequent.substitute(env)
                    if derived not in facts:
                        facts.add(derived)
                        changed = True
                        used = [str(a.substitute(env)) for a in rule.antecedents]
                        used_premises = [rule.source_id]
                        for u in used:
                            try:
                                atom_u = parse_atom(u)
                                if atom_u in proofs:
                                    used_premises.extend(proofs[atom_u].used_premises)
                            except Exception:
                                pass
                        proofs[derived] = ProofStep(
                            derived=str(derived),
                            rule_id=rule.source_id,
                            used=used,
                            used_premises=sorted(set(used_premises)),
                            note=f"Applied premise {rule.source_id}"
                        )
        return facts, proofs

    def prove_atom(self, target: Atom) -> ReasonResult:
        facts, proofs = self.closure()
        if target in facts:
            proof = self._trace_proof(target, proofs)
            used = sorted({p for st in proof for p in st.used_premises})
            return ReasonResult("Yes", proof, used, facts)
        neg_target = Atom(target.pred, target.args, not target.negated)
        if neg_target in facts:
            proof = self._trace_proof(neg_target, proofs)
            used = sorted({p for st in proof for p in st.used_premises})
            return ReasonResult("No", proof, used, facts)
        # If the query itself is negative, e.g. not can_transport_hazardous_materials(John),
        # support it by showing the positive target is blocked by an explicitly negated
        # requirement. This is needed for MCQ options such as "John cannot ...".
        if target.negated:
            positive_target = Atom(target.pred, target.args, False)
            blocked_pos = self._blocked_by_negated_requirement(positive_target, facts, seen=set())
            if blocked_pos:
                note = f"Positive target {positive_target} is blocked by negated requirement: {blocked_pos}"
                proof = [ProofStep(derived=str(target), rule_id=None, used=[blocked_pos], used_premises=[], note=note)]
                return ReasonResult("Yes", proof, [], facts, ["blocked_positive_target_by_negated_requirement"])

        blocked = self._blocked_by_negated_requirement(target, facts, seen=set())
        if blocked:
            note = f"Blocked by negated requirement while trying to prove {target}: {blocked}"
            proof = [ProofStep(derived=f"not {target}", rule_id=None, used=[blocked], used_premises=[], note=note)]
            return ReasonResult("No", proof, [], facts, ["blocked_by_negated_requirement"])
        return ReasonResult("Uncertain", [], [], facts, [f"Could not prove {target} or its negation."])

    def _unify_consequent(self, pattern: Atom, target: Atom) -> dict[str, str] | None:
        if pattern.pred != target.pred or pattern.negated != target.negated or len(pattern.args) != len(target.args):
            return None
        env: dict[str, str] = {}
        for pa, ta in zip(pattern.args, target.args):
            if is_var(pa):
                env[pa] = ta
            elif pa != ta:
                return None
        return env

    def _blocked_by_negated_requirement(self, target: Atom, facts: set[Atom], seen: set[Atom]) -> str | None:
        if target in seen:
            return None
        seen.add(target)
        direct_neg = Atom(target.pred, target.args, not target.negated)
        if direct_neg in facts:
            return str(direct_neg)
        for rule in self.kb.rules:
            env = self._unify_consequent(rule.consequent, target)
            if env is None:
                continue
            for ant in rule.antecedents:
                grounded = ant.substitute(env)
                neg = Atom(grounded.pred, grounded.args, not grounded.negated)
                if neg in facts:
                    return str(neg)
                nested = self._blocked_by_negated_requirement(grounded, facts, seen)
                if nested:
                    return nested
        return None

    def _trace_proof(self, target: Atom, proofs: dict[Atom, ProofStep], seen: set[Atom] | None = None) -> list[ProofStep]:
        seen = seen or set()
        if target in seen:
            return []
        seen.add(target)
        step = proofs.get(target)
        if not step:
            return []
        out: list[ProofStep] = []
        for u in step.used:
            try:
                out.extend(self._trace_proof(parse_atom(u), proofs, seen))
            except Exception:
                pass
        out.append(step)
        unique = []
        got = set()
        for s in out:
            if s.derived not in got:
                unique.append(s); got.add(s.derived)
        return unique
