from __future__ import annotations

import re
from difflib import SequenceMatcher

STATIC_CANONICAL_MAP = {
    'can_apply_for_collaborative_research_projects': 'can_apply_collaborative_projects',
    'can_apply_collaborative_research_projects': 'can_apply_collaborative_projects',
    'can_submit_research_proposals': 'can_submit_proposals',
    'can_access_restricted_archives': 'access_restricted_archives',
    'eligible_for_extended_library_access': 'extended_library_access',
    'teaches_for_at_least_5_years': 'taught_for_at_least_5_years',
    'can_teach_undergraduates': 'can_teach_undergrad',
    'can_teach_undergraduate_courses': 'can_teach_undergrad',
    'can_propose_courses': 'can_propose_new_courses',
    'contributes_original_perspectives': 'academic_contribution',
    'contribute_original_perspectives': 'academic_contribution',
}

PREFIX_STOPWORDS = ('can_', 'eligible_for_', 'qualifies_for_', 'has_', 'is_', 'are_')


def normalize_predicate_name_light(name: str) -> str:
    s = str(name or '').strip()
    s = s.replace('-', '_').replace(' ', '_')
    s = re.sub(r'[^A-Za-z0-9_]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_').lower()
    return STATIC_CANONICAL_MAP.get(s, s)


def predicate_signature_tokens(name: str) -> set[str]:
    n = normalize_predicate_name_light(name)
    for p in PREFIX_STOPWORDS:
        if n.startswith(p):
            n = n[len(p):]
    return {t for t in n.split('_') if t and t not in {'the', 'a', 'an', 'to', 'for', 'of', 'with'}}


def predicate_similarity(a: str, b: str) -> float:
    ca = normalize_predicate_name_light(a)
    cb = normalize_predicate_name_light(b)
    if ca == cb:
        return 1.0
    ta, tb = predicate_signature_tokens(ca), predicate_signature_tokens(cb)
    j = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0
    sm = SequenceMatcher(None, ca, cb).ratio()
    return max(j, sm)


def canonicalize_predicate(name: str, known: set[str] | None = None, threshold: float = 0.82) -> str:
    n = normalize_predicate_name_light(name)
    if not known:
        return n
    known_norm = {normalize_predicate_name_light(k): k for k in known}
    if n in known_norm:
        return normalize_predicate_name_light(known_norm[n])
    best = None
    best_score = 0.0
    for k in known_norm:
        score = predicate_similarity(n, k)
        if score > best_score:
            best, best_score = k, score
    return best if best is not None and best_score >= threshold else n
