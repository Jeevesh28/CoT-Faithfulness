from src.perturbations import (
    step_corruption, wrong_but_consistent, early_termination, step_shuffling,
)


STEPS = [
    "Step 1: Janet starts with 16 eggs per day.",
    "Step 2: She eats 3 and uses 4 for muffins.",
    "Step 3: 16 - 3 - 4 = 9 eggs remain.",
    "Step 4: At $2 each she earns $18.",
]


def test_step_corruption_changes_one_number():
    new, spans = step_corruption(STEPS, seed=1)
    assert len(spans) == 1
    # Span must land within the joined CoT.
    joined = "\n".join(new)
    a, b = spans[0]
    assert 0 <= a < b <= len(joined)
    assert new != STEPS


def test_step_corruption_preserves_last_step():
    # Repeatedly corrupt — should never touch the final step.
    for seed in range(10):
        new, spans = step_corruption(STEPS, seed=seed)
        assert new[-1] == STEPS[-1]


def test_wrong_but_consistent_replaces_entirely():
    donor = ["Step 1: foo", "Step 2: bar", "Answer: 7"]
    new, spans = wrong_but_consistent(STEPS, donor)
    assert new == donor
    assert spans and spans[0][0] == 0


def test_early_termination_half():
    new, _ = early_termination(STEPS, keep_frac=0.5)
    assert len(new) == 2
    new25, _ = early_termination(STEPS, keep_frac=0.25)
    assert len(new25) == 1


def test_step_shuffling_changes_order():
    new, _ = step_shuffling(STEPS, seed=1)
    assert new != STEPS
    assert sorted(new) == sorted(STEPS)
