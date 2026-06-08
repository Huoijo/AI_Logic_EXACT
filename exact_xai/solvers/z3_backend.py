from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..fol import Atom, KnowledgeBase, Rule, parse_fol_statement, parse_atom
from ..schemas import ProofStep
from ..reasoner import ReasonResult


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


@dataclass
class Z3Result:
    answer: str
    used_premises: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class Z3Backend:
    """Small finite-domain SMT backend for EXACT-style Horn rules.

    This is intentionally finite: all universal rules are grounded over constants
    observed in the KB. It is not a complete first-order prover, but it catches
    useful cases that the forward-chain proof engine misses, especially
    contraposition-style MCQ options.
    """
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        try:
            import z3  # type: ignore
            self.z3 = z3
            self.available = True
        except Exception:
            self.z3 = None
            self.available = False

    def _constants(self) -> list[str]:
        cs = sorted(self.kb.constants())
        return cs or ["GENERIC"]

    def _bool_for_atom(self, atom: Atom):
        z3 = self.z3
        key = f"{atom.pred}__{'__'.join(atom.args)}"
        b = z3.Bool(_safe_name(key))
        return z3.Not(b) if atom.negated else b

    def _ground_atom(self, atom: Atom, env: dict[str, str]) -> Atom:
        return atom.substitute(env)

    def _ground_rule_expr(self, rule: Rule, env: dict[str, str] | None = None):
        z3 = self.z3
        env = env or {}
        ants = [self._bool_for_atom(self._ground_atom(a, env)) for a in rule.antecedents]
        cons = self._bool_for_atom(self._ground_atom(rule.consequent, env))
        if ants:
            return z3.Implies(z3.And(*ants), cons)
        return cons

    def _add_kb(self, solver, track: bool = False):
        z3 = self.z3
        constants = self._constants()
        for f in self.kb.facts:
            expr = self._bool_for_atom(f)
            srcs = self.kb.fact_sources.get(f, []) or [0]
            if track:
                name = z3.Bool(f"p{srcs[0]}_fact_{abs(hash(str(f))) % 1000000}")
                solver.assert_and_track(expr, name)
            else:
                solver.add(expr)

        for rule in self.kb.rules:
            vars_ = sorted(rule.consequent.variables | set().union(*(a.variables for a in rule.antecedents)) if rule.antecedents else rule.consequent.variables)
            if vars_:
                import itertools
                for combo in itertools.product(constants, repeat=len(vars_)):
                    env = dict(zip(vars_, combo))
                    expr = self._ground_rule_expr(rule, env)
                    if track:
                        name = z3.Bool(f"p{rule.source_id}_rule_{abs(hash(str(rule.source_id)+str(env))) % 1000000}")
                        solver.assert_and_track(expr, name)
                    else:
                        solver.add(expr)
            else:
                expr = self._ground_rule_expr(rule)
                if track:
                    name = z3.Bool(f"p{rule.source_id}_rule_{abs(hash(str(rule.source_id))) % 1000000}")
                    solver.assert_and_track(expr, name)
                else:
                    solver.add(expr)

    def _premises_from_core(self, core: list[Any]) -> list[int]:
        out = set()
        for c in core:
            m = re.search(r"p(\d+)_", str(c))
            if m:
                pid = int(m.group(1))
                if pid:
                    out.add(pid)
        return sorted(out)

    def entails_atom(self, atom: Atom) -> Z3Result:
        if not self.available:
            return Z3Result("Uncertain", warnings=["z3_not_available"])
        z3 = self.z3
        # KB entails Q iff KB and not Q is unsat.
        s = z3.Solver()
        self._add_kb(s, track=True)
        s.assert_and_track(z3.Not(self._bool_for_atom(atom)), z3.Bool("negated_target"))
        if s.check() == z3.unsat:
            return Z3Result("Yes", self._premises_from_core(s.unsat_core()))

        # KB entails not Q iff KB and Q is unsat.
        s2 = z3.Solver()
        self._add_kb(s2, track=True)
        s2.assert_and_track(self._bool_for_atom(atom), z3.Bool("positive_target"))
        if s2.check() == z3.unsat:
            return Z3Result("No", self._premises_from_core(s2.unsat_core()))
        return Z3Result("Uncertain")

    def entails_rule(self, rule: Rule) -> Z3Result:
        if not self.available:
            return Z3Result("Uncertain", warnings=["z3_not_available"])
        z3 = self.z3
        constants = self._constants()
        vars_ = sorted(rule.consequent.variables | set().union(*(a.variables for a in rule.antecedents)) if rule.antecedents else rule.consequent.variables)
        # For each finite grounding, prove antecedents -> consequent.
        used = set()
        import itertools
        combos = itertools.product(constants, repeat=len(vars_)) if vars_ else [()]
        for combo in combos:
            env = dict(zip(vars_, combo))
            s = z3.Solver()
            self._add_kb(s, track=True)
            for ant in rule.antecedents:
                s.add(self._bool_for_atom(ant.substitute(env)))
            s.assert_and_track(z3.Not(self._bool_for_atom(rule.consequent.substitute(env))), z3.Bool("negated_rule_target"))
            if s.check() != z3.unsat:
                return Z3Result("Uncertain")
            used.update(self._premises_from_core(s.unsat_core()))
        return Z3Result("Yes", sorted(used))

    def prove_query_string(self, query: str) -> ReasonResult:
        q = query.strip()
        try:
            if "->" in q or "→" in q or q.lower().startswith("forall") or q.startswith("∀"):
                _, rules = parse_fol_statement(q, source_id=-1)
                if not rules:
                    return ReasonResult("Uncertain", warnings=[f"z3_query_not_rule: {q}"])
                zr = self.entails_rule(rules[0])
                proof = []
                if zr.answer == "Yes":
                    proof = [ProofStep(
                        derived=q,
                        rule_id=None,
                        used=[],
                        used_premises=zr.used_premises,
                        note="Verified by finite-domain Z3 entailment check",
                    )]
                return ReasonResult(zr.answer, proof, zr.used_premises, warnings=zr.warnings)
            atom = parse_atom(q)
            zr = self.entails_atom(atom)
            proof = []
            if zr.answer == "Yes":
                proof = [ProofStep(
                    derived=str(atom),
                    rule_id=None,
                    used=[],
                    used_premises=zr.used_premises,
                    note="Verified by Z3 entailment check",
                )]
            return ReasonResult(zr.answer, proof, zr.used_premises, warnings=zr.warnings)
        except Exception as e:
            return ReasonResult("Uncertain", warnings=[f"z3_query_error: {type(e).__name__}: {e}"])
