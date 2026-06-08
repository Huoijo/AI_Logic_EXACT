from exact_xai.dataset_utils import align_answers_to_questions


def test_answer_alignment_swaps_mcq_and_yesno():
    questions = [
        "Which is correct?\nA. Alpha\nB. Beta",
        "Does Sophia qualify?",
    ]
    aligned, warnings = align_answers_to_questions(questions, ["Yes", "A"])
    assert aligned == ["A", "Yes"]
    assert warnings
