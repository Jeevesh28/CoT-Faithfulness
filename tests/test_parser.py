from src.cot_parser import parse_cot, is_well_formed


def test_basic():
    txt = ("Step 1: Janet has 16 eggs per day.\n"
           "Step 2: She eats 3 and bakes with 4.\n"
           "Step 3: 16 - 3 - 4 = 9.\n"
           "Answer: 18")
    steps, ans = parse_cot(txt)
    assert len(steps) == 3
    assert ans == "18"
    assert is_well_formed(steps, ans)


def test_number_only_fallback():
    txt = "Let me think... the answer is 42."
    steps, ans = parse_cot(txt)
    assert ans == "42"


def test_no_answer_line():
    txt = "Step 1: foo\nStep 2: bar"
    steps, ans = parse_cot(txt)
    assert len(steps) == 2
    assert ans is None


def test_missing_step_tags_fallback():
    txt = "First we do one thing.\n\nThen we do another.\n\nAnswer: 7"
    steps, ans = parse_cot(txt)
    assert ans == "7"
    assert len(steps) >= 2  # fallback split by blank lines


def test_dotted_step_format():
    txt = "Step 1. foo\nStep 2. bar\nAnswer: 3"
    steps, ans = parse_cot(txt)
    assert len(steps) == 2
    assert ans == "3"
