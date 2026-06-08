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

def _match_fact(pattern: Atom, facts: set[Atom]) -> list[dict[str, str]]:
    envs = []
    for f in facts:
        if f.pred != pattern.pred or f.negated != pattern.negated or len(f.args) != len(pattern.args):
            continue
        env: dict[str, str] = {}
        ok = True
        for pa, fa in zip(pattern.args, f.args):
            if is_var(pa):
                if pa in env and env[pa] != fa:
                    ok = False; break
                env[pa] = fa
            elif pa != fa:
                ok = False; break
        if ok:
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
        return ReasonResult("Uncertain", [], [], facts, [f"Could not prove {target} or its negation."])

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
