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
