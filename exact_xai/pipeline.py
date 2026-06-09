from __future__ import annotations

from .schemas import AnswerRequest, AnswerResponse, ParsedQuestion
from .fol import parse_fol_premises, parse_atom
from .reasoner import Reasoner, ReasonResult
from .query_parser import (
    parse_question_rule_based, make_llm_prompt, parse_llm_json,
    parsed_target_to_atom, postprocess_parsed_question,
)
from .nl2logic import translate_nl_to_fol
from .explanation import proof_to_explanation
from .solvers.z3_backend import Z3Backend


class AnswerPipeline:
    def __init__(self, llm=None, input_mode: str = "auto", use_z3: bool = True):
        self.llm = llm
        self.input_mode = input_mode
        self.use_z3 = use_z3

    def build_kb(self, req: AnswerRequest):
        warnings = []
        raw = {}
        premises_fol = list(req.premises_fol or [])
        if self.input_mode == "nl":
            premises_fol = []
        if not premises_fol:
            translated = translate_nl_to_fol(req.premises_nl, req.question, self.llm)
            premises_fol = translated.premises_fol
            warnings.extend(["nl2logic:" + w for w in translated.warnings])
            raw["generated_premises_fol"] = premises_fol
            raw["nl2logic_raw"] = translated.raw
        if not premises_fol:
            warnings.append("no_logic_premises_available")
        kb = parse_fol_premises(premises_fol, req.premises_nl)
        return kb, warnings, raw

    def parse_question(self, req: AnswerRequest, kb) -> ParsedQuestion:
        if self.llm is not None:
            prompt = make_llm_prompt(req.question, kb)
            try:
                text = self.llm.generate(prompt, max_new_tokens=512, temperature=0.0)
                parsed = parse_llm_json(text)
                if parsed and (parsed.target or parsed.choices):
                    return postprocess_parsed_question(req.question, kb, parsed)
            except Exception:
                pass
        return parse_question_rule_based(req.question, kb)

    def prove_query(self, query: str | None, reasoner: Reasoner, z3_backend: Z3Backend | None) -> ReasonResult:
        if not query:
            return ReasonResult("Uncertain", warnings=["empty_query"])
        q = query.strip()
        # Rule/implication query: use Z3 finite entailment first.
        if "->" in q or "→" in q or q.lower().startswith("forall") or q.startswith("∀"):
            if z3_backend is not None:
                return z3_backend.prove_query_string(q)
            return ReasonResult("Uncertain", warnings=["rule_query_requires_z3"])

        atom = parsed_target_to_atom(q)
        if atom is None:
            return ReasonResult("Uncertain", warnings=[f"could_not_parse_query: {q}"])
        rr = reasoner.prove_atom(atom)
        if rr.answer == "Uncertain" and z3_backend is not None:
            zr = z3_backend.prove_query_string(q)
            if zr.answer != "Uncertain":
                return zr
        return rr

    def answer(self, req: AnswerRequest) -> AnswerResponse:
        kb, warnings, raw = self.build_kb(req)
        parsed = self.parse_question(req, kb)
        if parsed and parsed.raw.get("postprocess_warnings"):
            warnings.extend(parsed.raw.get("postprocess_warnings") or [])
        reasoner = Reasoner(kb)
        z3_backend = Z3Backend(kb) if self.use_z3 else None

        mode_parts = []
        mode_parts.append("nl2logic" if (self.input_mode == "nl" or not req.premises_fol) else "fol")
        mode_parts.append("qwen" if self.llm else "rule")
        mode_parts.append("symbolic")
        if self.use_z3:
            mode_parts.append("z3")
        mode = "_".join(mode_parts)

        if parsed.kind == "multiple_choice":
            option_results = {}
            choice_values = list(parsed.choices.values())
            if len(choice_values) != len(set(choice_values)):
                warnings.append("duplicate_choice_targets")

            for label, query in parsed.choices.items():
                option_results[label] = self.prove_query(query, reasoner, z3_backend)

            yes_options = [k for k, v in option_results.items() if v and v.answer == "Yes"]
            if len(yes_options) == 1:
                chosen = yes_options[0]
                rr = option_results[chosen]
                answer = chosen
            elif len(yes_options) > 1:
                def _choice_score(k: str):
                    rr_k = option_results[k]
                    return (len(rr_k.used_premises), len(rr_k.proof), k)
                chosen = sorted(yes_options, key=_choice_score)[0]
                rr = option_results[chosen]
                answer = chosen
                warnings.append(f"multiple_provable_options:{yes_options};selected:{chosen}")
            else:
                rr = None
                answer = "Uncertain"
                warnings.append("no_multiple_choice_option_provable")
            proof = rr.proof if rr else []
            used = rr.used_premises if rr else []
            explanation = proof_to_explanation("Yes" if answer != "Uncertain" else "Uncertain", proof, req.premises_nl, warnings)
            return AnswerResponse(
                id=req.id,
                answer=answer,
                mode=mode,
                parsed_question=parsed,
                used_premises=used,
                proof=proof,
                explanation=explanation,
                warnings=warnings,
                raw={**raw, "option_results": {k: (v.answer if v else None) for k, v in option_results.items()}},
            )

        rr = self.prove_query(parsed.target, reasoner, z3_backend)
        explanation = proof_to_explanation(rr.answer, rr.proof, req.premises_nl, rr.warnings + warnings)
        return AnswerResponse(
            id=req.id,
            answer=rr.answer,
            mode=mode,
            parsed_question=parsed,
            used_premises=rr.used_premises,
            proof=rr.proof,
            explanation=explanation,
            warnings=rr.warnings + warnings,
            raw=raw,
        )
