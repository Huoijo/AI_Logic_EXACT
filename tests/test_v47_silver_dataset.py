from exact_xai.silver_dataset import build_premise_translation_sample, build_question_parse_samples, split_by_family


def test_build_premise_translation_sample_schema():
    rec = {
        'premises-NL': ['If A then B.', 'Alice has A.'],
        'premises-FOL': ['ForAll(x, A(x) -> B(x))', 'A(Alice)'],
    }
    row = build_premise_translation_sample(rec, 0)
    assert row is not None
    assert row['sample_type'] == 'premise_nl2fol'
    assert row['target_json']['premises_fol'][0].startswith('ForAll')
    assert row['messages'][-1]['role'] == 'assistant'


def test_question_parse_sample_is_weak_not_answer_label():
    rec = {
        'premises-NL': ['If a student completes Course A, they can enroll in Course B.', 'David has completed Course A.'],
        'premises-FOL': ['ForAll(x, completed_course_a(x) -> enrolled_in_b(x))', 'completed_course_a(David)'],
        'questions': ['Can David enroll in Course B?'],
        'answers': ['Yes'],
    }
    rows = build_question_parse_samples(rec, 0)
    assert len(rows) == 1
    assert rows[0]['sample_type'] == 'question_parse'
    assert rows[0]['weight'] <= 1.0
    assert 'gold_answer' in rows[0]
    assert 'Answer: Yes' not in rows[0]['messages'][-1]['content']


def test_split_by_family_keeps_family_together():
    rows = [{'id': 'a1', 'family_id': 'A'}, {'id': 'a2', 'family_id': 'A'}, {'id': 'b1', 'family_id': 'B'}]
    train, valid = split_by_family(rows, valid_ratio=0.5, seed=0)
    fam_train = {r['family_id'] for r in train}
    fam_valid = {r['family_id'] for r in valid}
    assert fam_train.isdisjoint(fam_valid)
