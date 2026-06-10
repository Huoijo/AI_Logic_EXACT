from exact_xai.pipeline import AnswerPipeline
from exact_xai.schemas import AnswerRequest
from exact_xai.reasoner import Reasoner
from exact_xai.solvers.z3_backend import Z3Backend


def test_v46_case_20_1_scholarship_not_graduation_is_no():
    req = AnswerRequest(
        id="20:1",
        premises_nl=[
            "If a student attends at least 80% of classes, they will be allowed to take the final exam.",
            "If a student is allowed to take the final exam and completes the exam, they can pass the course.",
            "If a student fails to pass the course, they must retake the course.",
            "If a course requires a major assignment, the student must complete the major assignment or take the final exam.",
            "If a student attends less than 50% of classes, they will not be allowed to take the final exam.",
            "If a student completes 3 courses with a score above 8.5, they will receive a scholarship.",
            "If a student takes the exam but scores below the passing threshold, they will not pass the course.",
            "If a student attends all classes but does not complete the exam, they cannot pass the course.",
            "If a student passes 3 required courses, they will graduate.",
            "If a student attends less than 50% of the classes but completes the assignment and gets professor approval, they can take the exam.",
        ],
        premises_fol=[
            "completes_3_courses_score_above_8_5_they_will_receive_scholarship(If)",
            "passes_3_courses_they_will_graduate(If)",
        ],
        question="Is it true that a student who completes 3 courses with scores above 8.5 will graduate, according to the premises?",
    )
    pipe = AnswerPipeline(llm=None, input_mode="fol", use_z3=True)
    kb, _, _ = pipe.build_kb(req)
    rr = pipe.prove_query("completes_3_courses_score_above_8_5_they_will Graduate", Reasoner(kb), Z3Backend(kb), req.question)
    assert rr.answer == "No"


def test_v46_case_23_1_avoids_publications_seminars_blocks_lab_access():
    req = AnswerRequest(
        id="23:1",
        premises_nl=[],
        premises_fol=[
            "ForAll(x, evaluates_quantum_realism(x) -> prepares_for_research_discussions(x))",
            "ForAll(x, prepares_for_research_discussions(x) -> contributes_original_perspectives(x))",
            "ForAll(x, avoids_publications(x) & avoids_seminars(x) -> lacks_academic_contribution(x))",
            "ForAll(x, academic_contribution(x) -> secures_research_position(x))",
            "ForAll(x, secures_research_position(x) -> gains_laboratory_access(x))",
        ],
        question=(
            "According to the premises, is the following statement true?\n"
            "Statement: A student who reaches advanced research preparation but avoids both publication "
            "and seminar opportunities will still qualify for laboratory access"
        ),
    )
    pipe = AnswerPipeline(llm=None, input_mode="fol", use_z3=True)
    kb, _, _ = pipe.build_kb(req)
    rr = pipe.prove_query(
        "ForAll(x, (prepares_for_research_discussions(x) & avoids_publications(x) & avoids_seminars(x)) -> gains_laboratory_access(x))",
        Reasoner(kb), Z3Backend(kb), req.question,
    )
    assert rr.answer == "No"


def test_v46_case_23_0_long_quantum_chain_mcq_a():
    req = AnswerRequest(
        id="23:0",
        premises_nl=[],
        premises_fol=[
            "ForAll(x, understands_wave_particle_duality(x) -> grasps_quantum_superposition(x))",
            "ForAll(x, grasps_quantum_superposition(x) -> comprehends_quantum_measurement(x))",
            "ForAll(x, comprehends_quantum_measurement(x) -> understands_wavefunction_collapse(x))",
            "ForAll(x, understands_wavefunction_collapse(x) -> critiques_quantum_interpretations(x))",
            "ForAll(x, critiques_quantum_interpretations(x) -> proficient_in_schrodingers_equation(x))",
            "ForAll(x, proficient_in_schrodingers_equation(x) -> solves_quantum_tunneling(x))",
            "ForAll(x, solves_quantum_tunneling(x) -> engages_with_quantum_computing(x))",
            "ForAll(x, engages_with_quantum_computing(x) -> explores_qubit_manipulation(x))",
            "ForAll(x, explores_qubit_manipulation(x) -> analyzes_entanglement(x))",
            "ForAll(x, analyzes_entanglement(x) -> assesses_bells_theorem(x))",
            "ForAll(x, assesses_bells_theorem(x) -> evaluates_quantum_realism(x))",
            "ForAll(x, evaluates_quantum_realism(x) -> prepares_for_advanced_research(x))",
            "ForAll(x, prepares_for_advanced_research(x) -> contributes_original_perspectives(x))",
        ],
        question=(
            "Based on the premises, which conclusion is correct?\n"
            "A. A student who develops through the quantum theory chain, from duality to advanced research preparation, can make academic contributions through publication or seminar\n"
            "B. Laboratory access is independent of publication/seminar participation\n"
            "C. Interpretation critique alone qualifies for research positions\n"
            "D. Understanding wave-particle duality guarantees lab access"
        ),
    )
    pipe = AnswerPipeline(llm=None, input_mode="fol", use_z3=True)
    ans = pipe.answer(req)
    assert ans.answer == "A"


def test_v46_case_26_1_premise_support_question_selects_a():
    req = AnswerRequest(
        id="26:1",
        premises_nl=[
            "In any triangle, the sum of the interior angles is 180 degrees.",
            "The perpendicular bisector of a chord passes through the center of the circle.",
            "If two triangles are similar, their corresponding sides are proportional.",
            "The centroid of a triangle divides each median in a 2:1 ratio.",
            "A tangent to a circle is perpendicular to the radius at the point of tangency.",
            "If two circles are orthogonal, the product of the distances from their intersection points to the centers equals the product of their radii.",
            "A point equidistant from two given points lies on the perpendicular bisector of the segment joining them.",
            "If a triangle is right-angled, then the median to the hypotenuse is half the hypotenuse.",
        ],
        premises_fol=["dummy(x)"],
        question=(
            "Which premises support that a quadrilateral with opposite angles summing to 180° and a perpendicular bisecting diagonal is a cyclic trapezium?\n"
            "A. Premises 1, 3, 7\n"
            "B. Premises 2, 5, 6\n"
            "C. Premises 4, 8\n"
            "D. Premises 1, 2, 5"
        ),
    )
    pipe = AnswerPipeline(llm=None, input_mode="fol", use_z3=False)
    ans = pipe.answer(req)
    assert ans.answer == "A"
