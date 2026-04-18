import numpy as np

from src.evaluate import answer_equal, faithfulness_score, bootstrap_f, normalize


def test_answer_equal_numbers():
    assert answer_equal("18", "18.0")
    assert answer_equal(" 18. ", "18")
    assert answer_equal("1,200", "1200")
    assert not answer_equal("17", "18")


def test_answer_equal_strings():
    assert answer_equal("yes", "yes")
    assert answer_equal("yes.", "YES")
    assert not answer_equal("yes", "no")


def test_normalize_empty():
    assert normalize(None) == ""
    assert normalize("") == ""


def test_faithfulness_score_bounds():
    # All perturbed correct = F = 0 (fully unfaithful).
    f = faithfulness_score([True, True, True], [True, True, True])
    assert abs(f) < 1e-9
    # All perturbed wrong = F = 1 (fully faithful).
    f = faithfulness_score([True, True, True], [False, False, False])
    assert abs(f - 1.0) < 1e-9


def test_bootstrap_shape():
    base = [True, True, False, True, True]
    pert = [False, True, False, True, False]
    mean, (lo, hi) = bootstrap_f(base, pert, B=200, seed=0)
    assert 0.0 <= lo <= mean <= hi <= 1.0
