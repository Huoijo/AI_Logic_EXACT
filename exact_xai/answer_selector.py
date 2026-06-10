from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

POSITIVE_PREFIXES = (
    'can_', 'eligible_', 'eligible_for_', 'qualifies_', 'qualifies_for_', 'authorized_',
    'receives_', 'enhances_', 'scholarship_', 'academic_contribution', 'gains_',
)
NEGATIVE_TEXT_MARKERS = (
    'cannot', "can't", 'not ', 'without', 'needs ', 'must ', 'lacks', 'insufficient', 'cannot',
)


@dataclass
class Selection:
    answer: str
    warnings: list[str]
    scores: dict[str, float]


def _pred_from_query(query: str) -> str:
    q = str(query or '').strip().lower()
    m = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', q)
    return m.group(1).lower() if m else q


def _tokens(text: str) -> set[str]:
    return {t for t in re.sub(r'[^a-z0-9_]+', ' ', str(text).lower()).split() if len(t) > 2}


def score_option(label: str, query: str, option_text: str, option_answer: str, question: str = '', used_premises: list[int] | None = None) -> float:
    score = 0.0
    if option_answer == 'Yes':
        score += 100.0
    elif option_answer == 'No':
        score -= 40.0
    else:
        score -= 10.0

    pred = _pred_from_query(query)
    if pred.startswith(POSITIVE_PREFIXES):
        score += 10.0
    if str(query or '').strip().lower().startswith('not '):
        score -= 15.0

    qtok = _tokens(question)
    otok = _tokens(option_text) | _tokens(query)
    if qtok and otok:
        score += 8.0 * (len(qtok & otok) / max(1, len(qtok | otok)))

    option_l = str(option_text or '').lower()
    if any(x in option_l for x in NEGATIVE_TEXT_MARKERS):
        score -= 12.0
    if ' but ' in option_l and any(x in option_l for x in ('cannot', "can't", 'not ')):
        score -= 18.0

    used = len(used_premises or [])
    if 'fewest premise' in str(question).lower():
        score -= used * 4.0
    else:
        score += min(used, 6) * 0.5
    return score


def select_mcq(question: str, choices_text: dict[str, str], choices_fol: dict[str, str], option_results: dict[str, Any]) -> Selection:
    scores: dict[str, float] = {}
    for label, query in choices_fol.items():
        rr = option_results.get(label)
        ans = getattr(rr, 'answer', rr if isinstance(rr, str) else None)
        used = getattr(rr, 'used_premises', []) if rr is not None else []
        scores[label] = score_option(label, query, choices_text.get(label, ''), str(ans), question, used)
    if not scores:
        return Selection('Uncertain', ['answer_selector_no_options'], {})
    best = max(scores, key=lambda k: (scores[k], -ord(k[0]) if k else 0))
    warnings = [f'proof_aware_selector:{best}']
    return Selection(best, warnings, scores)
