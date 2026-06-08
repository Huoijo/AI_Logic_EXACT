from exact_xai.fol import parse_fol_premises, parse_atom
from exact_xai.reasoner import Reasoner


def test_simple_chain():
    kb = parse_fol_premises([
        "ForAll(x, A(x) -> B(x))",
        "ForAll(x, B(x) -> C(x))",
        "A(Sophia)",
    ])
    rr = Reasoner(kb).prove_atom(parse_atom("C(Sophia)"))
    assert rr.answer == "Yes"
