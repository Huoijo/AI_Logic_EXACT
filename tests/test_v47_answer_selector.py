from types import SimpleNamespace
from exact_xai.answer_selector import select_mcq


def rr(answer, used=None):
    return SimpleNamespace(answer=answer, used_premises=used or [])


def test_selector_prefers_direct_positive_capability_over_negative_mixed_option():
    question = 'Based on status, which statement is correct about booking training?'
    choices_text = {
        'A': "Alex can use equipment but can't book training",
        'B': 'Alex can book personal training sessions',
    }
    choices_fol = {
        'A': 'can_use_equipment(Alex) & not can_book_training(Alex)',
        'B': 'can_book_training(Alex)',
    }
    sel = select_mcq(question, choices_text, choices_fol, {'A': rr('Yes'), 'B': rr('Yes')})
    assert sel.answer == 'B'


def test_selector_returns_uncertain_without_options():
    sel = select_mcq('q', {}, {}, {})
    assert sel.answer == 'Uncertain'
