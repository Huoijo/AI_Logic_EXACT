from exact_xai.fol_repair import repair_fol_string
from exact_xai.fol import parse_fol_premises, parse_atom
from exact_xai.reasoner import Reasoner


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


def test_numeric_predicate_threshold_entails_lower_threshold():
    kb = parse_fol_premises([
        "ForAll(x, completing_500_clinical_hours(x) -> advanced_practice_certification(x))",
        "ForAll(x, (registered_nurse(x) & advanced_practice_certification(x)) -> authorized_to_prescribe_medication(x))",
        "completing_600_clinical_hours(John)",
        "registered_nurse(John)",
    ])
    rr = Reasoner(kb).prove_atom(parse_atom("authorized_to_prescribe_medication(John)"))
    assert rr.answer == "Yes"


def test_repeated_program_suffix_collapse():
    from exact_xai.fol_repair import normalize_predicate_name, repair_fol_string

    assert normalize_predicate_name(
        "qualifies_for_graduate_fellowship_program_program_program"
    ) == "qualifies_for_graduate_fellowship_program"
    assert repair_fol_string(
        "qualifies_for_graduate_fellowship_program_program(John)"
    ) == "qualifies_for_graduate_fellowship_program(John)"


def test_yes_no_if_all_conditional_not_tautology():
    from exact_xai.query_parser import parse_question_rule_based

    kb = parse_fol_premises([
        "ForAll(x, well_structured_project(x) -> optimized_project(x))",
        "ForAll(x, python_project(x) -> well_structured_project(x))",
    ])
    q = (
        "Does it follow that if all Python projects are well-structured, "
        "then all Python projects are optimized, according to the premises?"
    )
    parsed = parse_question_rule_based(q, kb)
    assert parsed.kind == "yes_no"
    assert parsed.target == "ForAll(x, well_structured_project(x) -> optimized_project(x))"


def test_john_fellowship_option_postprocess():
    from exact_xai.query_parser import parse_question_rule_based

    kb = parse_fol_premises([
        "ForAll(x, completed_all_required_courses(x) -> eligible_for_graduation(x))",
        "ForAll(x, (eligible_for_graduation(x) & gpa_above_3_5(x)) -> graduated_with_honors(x))",
        "ForAll(x, (graduated_with_honors(x) & completed_thesis(x)) -> receives_academic_distinction(x))",
        "ForAll(x, receives_academic_distinction(x) -> qualifies_for_graduate_fellowship_program(x))",
        "completed_all_required_courses(John)",
        "gpa_above_3_5(John)",
        "completed_thesis(John)",
    ])
    q = """Based on the above premises, which conclusion logically follows?
A. John qualifies for the graduate fellowship program
B. John needs faculty recommendation for the fellowship
C. John must complete an internship to qualify
D. John’s GPA is insufficient for honors"""
    parsed = parse_question_rule_based(q, kb)
    assert parsed.choices["A"] == "qualifies_for_graduate_fellowship_program(John)"
    assert parsed.choices["B"] == "received_faculty_recommendation(John)"
    assert parsed.choices["C"] == "completed_internship(John)"
    assert parsed.choices["D"] == "not gpa_above_3_5(John)"
