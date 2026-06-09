from exact_xai.fol_repair import repair_fol_string, normalize_predicate_name
from exact_xai.fol import parse_fol_premises, parse_atom
from exact_xai.reasoner import Reasoner
from exact_xai.query_parser import parse_question_rule_based, postprocess_parsed_question, parse_llm_json
from exact_xai.nl2logic import contextual_repair_with_nl


def test_method_syntax_repair_chain():
    fol = [
        repair_fol_string("ForAll(x, x.completed_course_a -> x.enrolled_in_b)"),
        repair_fol_string("David.completed_course_a"),
    ]
    assert fol == ["ForAll(x, completed_course_a(x) -> enrolled_in_b(x))", "completed_course_a(David)"]
    kb = parse_fol_premises(fol)
    assert Reasoner(kb).prove_atom(parse_atom("enrolled_in_b(David)")).answer == "Yes"


def test_negated_requirement_blocks_downstream():
    kb = parse_fol_premises([
        "ForAll(x, (can_transport_standard_goods(x) & completed_hazmat_training(x) & received_safety_endorsement(x)) -> can_transport_hazardous_materials(x))",
        "ForAll(x, (can_transport_hazardous_materials(x) & has_interstate_permit(x)) -> can_cross_state_lines_hazardous(x))",
        "can_transport_standard_goods(John)",
        "completed_hazmat_training(John)",
        "not_received_safety_endorsement(John)",
        "has_interstate_permit(John)",
    ])
    rr = Reasoner(kb).prove_atom(parse_atom("can_cross_state_lines_hazardous(John)"))
    assert rr.answer == "No"


def test_negative_mcq_option_supported_by_blocked_requirement():
    kb = parse_fol_premises([
        "ForAll(x, (can_transport_standard_goods(x) & completed_hazmat_training(x) & received_safety_endorsement(x)) -> can_transport_hazardous_materials(x))",
        "can_transport_standard_goods(John)",
        "completed_hazmat_training(John)",
        "not_received_safety_endorsement(John)",
    ])
    rr = Reasoner(kb).prove_atom(parse_atom("not can_transport_hazardous_materials(John)"))
    assert rr.answer == "Yes"


def test_numeric_predicate_threshold_entails_lower_threshold():
    kb = parse_fol_premises([
        "ForAll(x, completing_500_clinical_hours(x) -> advanced_practice_certification(x))",
        "ForAll(x, (registered_nurse(x) & advanced_practice_certification(x)) -> authorized_to_prescribe_medication(x))",
        "completing_600_clinical_hours(John)",
        "registered_nurse(John)",
    ])
    rr = Reasoner(kb).prove_atom(parse_atom("authorized_to_prescribe_medication(John)"))
    assert rr.answer == "Yes"


def test_repeated_program_suffix_collapses():
    assert normalize_predicate_name("qualifies_for_graduate_fellowship_program_program_program") == "qualifies_for_graduate_fellowship_program"
    assert repair_fol_string("qualifies_for_graduate_fellowship_program_program(John)") == "qualifies_for_graduate_fellowship_program(John)"


def test_conditional_yes_no_target_not_tautology():
    kb = parse_fol_premises([
        "ForAll(x, well_structured_project(x) -> optimized_project(x))",
        "Exists(x, optimized_project(x))",
    ])
    q = "Does it follow that if all Python projects are well-structured, then all Python projects are optimized, according to the premises?"
    parsed = parse_question_rule_based(q, kb)
    assert parsed.target == "ForAll(x, well_structured_project(x) -> optimized_project(x))"


def test_john_fellowship_option_atom():
    kb = parse_fol_premises([
        "ForAll(x, received_academic_distinction(x) -> qualifies_for_graduate_fellowship_program(x))",
        "completed_thesis(John)",
    ])
    q = "Based on the above premises, which conclusion logically follows?\nA. John qualifies for the graduate fellowship program\nB. John needs faculty recommendation for the fellowship\nC. John must complete an internship to qualify\nD. John’s GPA is insufficient for honors"
    llm = parse_llm_json('{"kind":"multiple_choice","choices":{"A":"qualifies_for_graduate_fellowship_program(John)","B":"not completed_all_required_courses(John) -> not qualifies_for_graduate_fellowship_program(John)","C":"not completed_thesis(John) -> not qualifies_for_graduate_fellowship_program(John)","D":"not gpa_above_3_5(John) -> not qualifies_for_graduate_fellowship_program(John)"}}')
    parsed = postprocess_parsed_question(q, kb, llm)
    assert parsed.choices["A"] == "qualifies_for_graduate_fellowship_program(John)"
    assert parsed.choices["C"] == "completed_internship(John)"
    assert parsed.choices["D"] == "not gpa_above_3_5(John)"


def test_collaborative_research_alias_chain():
    kb = parse_fol_premises([
        "ForAll(x, taught_for_at_least_5_years(x) -> extended_library_access(x))",
        "ForAll(x, (extended_library_access(x) & published_at_least_one_paper(x)) -> access_restricted_archives(x))",
        "ForAll(x, (access_restricted_archives(x) & completed_research_ethics_training(x)) -> can_submit_proposals(x))",
        "ForAll(x, (can_submit_proposals(x) & has_departmental_endorsement(x)) -> can_apply_collaborative_projects(x))",
        "teaches_for_at_least_5_years(John)",
        "published_at_least_one_paper(John)",
        "completed_research_ethics_training(John)",
        "has_departmental_endorsement(John)",
    ])
    assert Reasoner(kb).prove_atom(parse_atom("can_apply_collaborative_projects(John)")).answer == "Yes"


def test_curriculum_cross_entity_contextual_repair():
    nl = [
        "If a curriculum is well-structured and has exercises, it enhances student engagement.",
        "If a curriculum enhances student engagement and provides access to advanced resources, it enhances critical thinking.",
        "If a faculty prioritizes pedagogical training and curriculum development, the curriculum is well-structured.",
        "The faculty prioritizes pedagogical training and curriculum development.",
        "The curriculum has practical exercises.",
        "The curriculum provides access to advanced resources.",
    ]
    fol = [
        "ForAll(x, well_structured_curriculum(x) & has_exercises(x) -> enhances_engagement(x))",
        "ForAll(x, enhances_engagement(x) & provides_access_to_advanced_resources(x) -> enhances_critical_thinking(x))",
        "ForAll(x, prioritizes_pedagogical_training(x) & develops_curriculum(x) -> well_structured_curriculum(x))",
        "prioritizes_pedagogical_training(faculty)",
        "has_exercises(curriculum)",
        "provides_access_to_advanced_resources(curriculum)",
    ]
    fixed = contextual_repair_with_nl(nl, fol)
    # premise 4 should become a conjunctive fact and premise 3 should ground the curriculum consequent
    assert "well_structured_curriculum(curriculum)" in fixed[2]
    assert fixed[3] == "prioritizes_pedagogical_training(faculty) & develops_curriculum(faculty)"
    kb = parse_fol_premises(fixed)
    assert Reasoner(kb).prove_atom(parse_atom("enhances_critical_thinking(curriculum)")).answer == "Yes"


def test_david_needs_to_pass_course_b_first_is_negative():
    kb = parse_fol_premises([
        "ForAll(x, enrolled_in_b(x) & passed_b(x) -> enrolled_in_c(x))",
        "ForAll(x, enrolled_in_c(x) -> eligible_for_internship(x))",
        "enrolled_in_b(David) & passed_b(David)",
    ])
    q = "Based on the prerequisites, what is David’s current eligibility status?\nA. Eligible for Course C but not the internship\nB. Eligible for the internship program\nC. Needs to pass Course B first\nD. Only eligible for Course B"
    parsed = parse_question_rule_based(q, kb)
    assert parsed.choices["B"] == "eligible_for_internship(David)"
    assert parsed.choices["C"] == "not passed_b(David)"
    assert parsed.choices["D"] == "not eligible_for_internship(David)"


def test_v43_well_structured_project_alias_keeps_python_conditional_target():
    kb = parse_fol_premises([
        "ForAll(x, well_structured_project(x) -> optimized_project(x))",
        "Exists(x, optimized_project(x))",
    ])
    q = "Does it follow that if all Python projects are well-structured, then all Python projects are optimized, according to the premises?"
    parsed = parse_question_rule_based(q, kb)
    assert parsed.target == "ForAll(x, well_structured_project(x) -> optimized_project(x))"


def test_v43_hazmat_but_cannot_option_does_not_duplicate_negative_target():
    kb = parse_fol_premises([
        "ForAll(x, can_transport_standard_goods(x) & completed_hazmat_training(x) & received_safety_endorsement(x) -> can_transport_hazardous_materials(x))",
        "not_received_safety_endorsement(John)",
    ])
    q = "Based on the premises, what can we conclude about John’s qualifications?\nA. John can transport hazardous materials but cannot cross state lines\nB. John can cross state lines with hazardous cargo\nC. John cannot transport hazardous materials\nD. John is not qualified to transport any kind of goods"
    parsed = parse_question_rule_based(q, kb)
    assert parsed.choices["A"] == "can_transport_hazardous_materials(John) & not can_cross_state_lines_hazardous(John)"
    assert parsed.choices["C"] == "not can_transport_hazardous_materials(John)"
    assert parsed.choices["A"] != parsed.choices["C"]


def test_v431_fewest_premises_prefers_direct_contrapositive_cost():
    from exact_xai.pipeline import _direct_rule_or_contraposition_cost
    kb = parse_fol_premises([
        "ForAll(x, well_tested_code(x) -> optimized_project(x))",
        "ForAll(x, well_structured_project(x) -> optimized_project(x))",
        "ForAll(x, not_well_structured(x) -> not_follow_pep8(x))",
    ])
    assert _direct_rule_or_contraposition_cost(
        "ForAll(x, not optimized_project(x) -> not well_tested(x))", kb
    ) == 1
    assert _direct_rule_or_contraposition_cost(
        "ForAll(x, not optimized_project(x) -> not follow_pep8(x))", kb
    ) is None
