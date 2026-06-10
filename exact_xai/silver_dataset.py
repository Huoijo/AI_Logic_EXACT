from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> Any:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    rows: list[dict[str, Any]] = []
    if not p.exists():
        return rows
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def unwrap_dataset(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ('records', 'data', 'examples', 'train'):
            if isinstance(obj.get(key), list):
                return [x for x in obj[key] if isinstance(x, dict)]
    raise ValueError('Dataset must be a list or a dict containing records/data/examples/train list.')


def get_premises_nl(record: dict[str, Any]) -> list[str]:
    return as_list(record.get('premises-NL') or record.get('premises_nl') or record.get('nl_premises') or record.get('premises'))


def get_premises_fol(record: dict[str, Any]) -> list[str]:
    return as_list(record.get('premises-FOL') or record.get('premises_fol') or record.get('fol_premises'))


def get_questions(record: dict[str, Any]) -> list[str]:
    return as_list(record.get('questions') or record.get('question'))


def get_answers(record: dict[str, Any]) -> list[str]:
    return as_list(record.get('answers') or record.get('answer'))


def stable_family_id(record: dict[str, Any], idx: int) -> str:
    rid = record.get('id') or record.get('record_id') or record.get('uid')
    return str(rid) if rid is not None else str(idx)


def format_numbered(items: list[str]) -> str:
    return '\n'.join(f'{i}. {x}' for i, x in enumerate(items, 1))


def extract_choices(question: str) -> dict[str, str]:
    # Keep a local implementation so silver building works even if query_parser changes.
    choice_re = re.compile(r'(?m)^\s*([A-D])\.\s*(.+?)(?=\n\s*[A-D]\.\s*|\Z)', re.S)
    return {m.group(1): ' '.join(m.group(2).split()) for m in choice_re.finditer(question or '')}


def make_chat_text(messages: list[dict[str, str]]) -> str:
    # Tokenizer-independent fallback; train script will re-render with chat template if desired.
    parts = []
    for m in messages:
        parts.append(f"<{m['role']}>\n{m['content']}")
    return '\n\n'.join(parts)


def build_premise_translation_sample(record: dict[str, Any], idx: int) -> dict[str, Any] | None:
    premises_nl = get_premises_nl(record)
    premises_fol = get_premises_fol(record)
    if not premises_nl or not premises_fol:
        return None
    system = (
        'You are an NL-to-FOL semantic parser for EXACT. '
        'Translate natural-language premises into First-Order Logic. '
        'Return strict JSON only.'
    )
    user = (
        'Convert the following premises into FOL while preserving premise order.\n\n'
        f'Premises:\n{format_numbered(premises_nl)}\n\n'
        'Return schema: {"premises_fol": ["..."]}'
    )
    assistant = json.dumps({'premises_fol': premises_fol}, ensure_ascii=False, indent=2)
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
        {'role': 'assistant', 'content': assistant},
    ]
    return {
        'id': f'{stable_family_id(record, idx)}:premises',
        'family_id': stable_family_id(record, idx),
        'sample_type': 'premise_nl2fol',
        'confidence': 1.0,
        'weight': 1.0,
        'messages': messages,
        'text': make_chat_text(messages),
        'target_json': {'premises_fol': premises_fol},
        'source': 'gold_premises_fol',
    }


def build_question_parse_samples(record: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    """Build weak parser samples using the current rule-based parser when possible.

    This deliberately avoids using gold answer as the desired output. The model learns
    the JSON schema and predicate style; the symbolic engine remains responsible for answers.
    """
    premises_nl = get_premises_nl(record)
    premises_fol = get_premises_fol(record)
    questions = get_questions(record)
    answers = get_answers(record)
    if not premises_nl or not premises_fol or not questions:
        return []

    # Import lazily so this file is also usable outside the full package.
    try:
        from exact_xai.fol import parse_fol_premises
        from exact_xai.query_parser import parse_question_rule_based
    except Exception:
        parse_fol_premises = None
        parse_question_rule_based = None

    kb = None
    if parse_fol_premises is not None:
        try:
            kb = parse_fol_premises(premises_fol, premises_nl)
        except Exception:
            kb = None

    rows: list[dict[str, Any]] = []
    family = stable_family_id(record, idx)
    for q_idx, question in enumerate(questions):
        choices_nl = extract_choices(question)
        kind = 'multiple_choice' if choices_nl else 'yes_no'
        target_json: dict[str, Any] = {
            'kind': kind,
            'premises_fol': premises_fol,
            'target': None,
            'choices': {},
        }
        confidence = 0.35
        source = 'schema_only'
        if kb is not None and parse_question_rule_based is not None:
            try:
                parsed = parse_question_rule_based(question, kb)
                target_json['kind'] = parsed.kind
                target_json['target'] = parsed.target
                target_json['choices'] = parsed.choices or {}
                # Treat rule-based targets as weak labels: useful for schema/style, not truth.
                confidence = 0.60 if (parsed.target or parsed.choices) else 0.25
                source = 'rule_based_parser_weak'
            except Exception:
                pass
        system = (
            'You are an NL-to-FOL semantic parser for EXACT. '
            'Parse the question into strict JSON. Do not answer the question.'
        )
        user = (
            f'Premises NL:\n{format_numbered(premises_nl)}\n\n'
            f'Premises FOL:\n{format_numbered(premises_fol)}\n\n'
            f'Question:\n{question}\n\n'
            'Return schema: {"kind": "yes_no|multiple_choice", "target": "..." or null, "choices": {"A":"..."}}'
        )
        assistant = json.dumps({k: target_json[k] for k in ('kind', 'target', 'choices')}, ensure_ascii=False, indent=2)
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': assistant},
        ]
        rows.append({
            'id': f'{family}:{q_idx}:question',
            'case_id': f'{idx}:{q_idx}',
            'family_id': family,
            'question_index': q_idx,
            'sample_type': 'question_parse',
            'confidence': confidence,
            'weight': confidence,
            'gold_answer': answers[q_idx] if q_idx < len(answers) else None,
            'messages': messages,
            'text': make_chat_text(messages),
            'target_json': target_json,
            'source': source,
        })
    return rows


def split_by_family(rows: list[dict[str, Any]], valid_ratio: float = 0.15, seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row.get('family_id', row.get('id', 'unknown'))), []).append(row)
    families = sorted(by_family)
    rng = random.Random(seed)
    rng.shuffle(families)
    if len(families) <= 1 or valid_ratio <= 0:
        return rows, []
    n_valid = max(1, int(round(len(families) * valid_ratio)))
    valid_families = set(families[:n_valid])
    train, valid = [], []
    for fam, fam_rows in by_family.items():
        (valid if fam in valid_families else train).extend(fam_rows)
    return train, valid
