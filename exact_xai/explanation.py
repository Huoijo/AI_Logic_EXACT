from __future__ import annotations

from .schemas import ProofStep


def proof_to_explanation(
    answer: str,
    proof: list[ProofStep],
    premises_nl: list[str] | None = None,
    warnings: list[str] | None = None,
    explanation_style: str = "short",
) -> str:
    """Turn a proof trace into a compact explanation.

    short: concise enough for smoke tests and API output.
    verbose: one sentence per proof step.
    """
    premises_nl = premises_nl or []
    warnings = warnings or []

    if answer == "Uncertain":
        base = "The system cannot determine the answer from the given premises."
        if warnings:
            base += " " + " ".join(warnings)
        return base

    if not proof:
        return f"The answer is {answer}, but no detailed proof trace was produced."

    if explanation_style == "verbose":
        parts = []
        for step in proof:
            if step.rule_id is None:
                parts.append(f"Given: {step.derived}.")
            else:
                src = ""
                if 1 <= step.rule_id <= len(premises_nl):
                    src = f" Premise {step.rule_id} says: {premises_nl[step.rule_id-1]}"
                used = f" from {', '.join(step.used)}" if step.used else ""
                parts.append(f"Using premise {step.rule_id}{used}, derive {step.derived}.{src}")
        return " ".join(parts)

    given = [s.derived for s in proof if s.rule_id is None]
    derived_steps = [s for s in proof if s.rule_id is not None]

    pieces = []
    if given:
        if len(given) <= 3:
            pieces.append("Given " + ", ".join(given) + ".")
        else:
            pieces.append("Using the given facts plus the cited rules.")

    for step in derived_steps:
        used = f" from {', '.join(step.used)}" if step.used else ""
        premise = f" by premise {step.rule_id}" if step.rule_id is not None else ""
        pieces.append(f"Derive {step.derived}{used}{premise}.")

    if warnings:
        pieces.append("Warnings: " + " ".join(warnings))

    return " ".join(pieces)
