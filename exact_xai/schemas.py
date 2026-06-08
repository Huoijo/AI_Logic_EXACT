from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

AnswerLabel = Literal["Yes", "No", "Uncertain"]

class AnswerRequest(BaseModel):
    id: str | None = None
    premises_nl: list[str] = Field(default_factory=list)
    premises_fol: list[str] = Field(default_factory=list)
    question: str
    question_type: str | None = None
    choices: dict[str, str] | None = None
    gold_answer: str | None = None

class ParsedQuestion(BaseModel):
    kind: Literal["yes_no", "multiple_choice", "open"] = "yes_no"
    target: str | None = None
    choices: dict[str, str] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)
    parser: str = "unknown"

class ProofStep(BaseModel):
    derived: str
    rule_id: int | None = None
    used: list[str] = Field(default_factory=list)
    used_premises: list[int] = Field(default_factory=list)
    note: str = ""

class AnswerResponse(BaseModel):
    id: str | None = None
    answer: str
    mode: str
    parsed_question: ParsedQuestion | None = None
    used_premises: list[int] = Field(default_factory=list)
    proof: list[ProofStep] = Field(default_factory=list)
    explanation: str
    warnings: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

class BatchRequest(BaseModel):
    records: list[AnswerRequest]

class BatchResponse(BaseModel):
    results: list[AnswerResponse]
