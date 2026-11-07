from metrics import (ABSTAIN, abstention_rate, accuracy, bootstrap_macro_f1,
                     coverage_weighted_accuracy, groundedness, macro_f1,
                     output_consistency, per_class_metrics, wilson_interval)


def test_perfect_and_constant_predictions():
    y = ["A", "A", "B", "B"]
    assert macro_f1(y, y) == 1.0
    assert accuracy(y, ["A"] * 4) == 0.5
    assert macro_f1(y, ["A"] * 4) < 0.5      # accuracy hides the failure


def test_macro_f1_is_class_balanced():
    y = ["A"] * 9 + ["B"]
    assert macro_f1(y, ["A"] * 10) < 0.55
    m = per_class_metrics(y, ["A"] * 10)
    assert m["A"]["recall"] == 1.0 and m["B"]["recall"] == 0.0


def test_bootstrap_interval_is_ordered_and_in_range():
    y = ["A", "B", "A", "B", "C", "A", "B", "C"]
    p = ["A", "B", "A", "A", "C", "A", "B", "B"]
    ci = bootstrap_macro_f1(y, p, n_boot=80, seed=7)
    assert 0 <= ci["ci_low"] <= ci["ci_high"] <= 1
    assert ci["n_boot"] == 80 and ci["ci_level"] == 0.95


def test_groundedness_requires_verbatim_evidence():
    text = "My credit report shows an account that is not mine."
    assert groundedness("an account that is not mine", text) == 1.0
    assert groundedness("", text) == 0.0
    assert groundedness("the bank stole my dog in 1999", text) < 0.5


def test_abstention_and_coverage():
    y = ["A", "B", "C"]
    p = ["A", ABSTAIN, ABSTAIN]
    assert abstention_rate(p) == 2 / 3
    c = coverage_weighted_accuracy(y, p)
    assert c["n_answered"] == 1 and abs(c["coverage"] - 1 / 3) < 1e-6


def test_output_consistency_and_wilson_bounds():
    assert output_consistency([["A", "B"], ["A", "B"]])["all_runs_identical_rate"] == 1.0
    assert output_consistency([["A", "B"], ["A", "C"]])["mean_pairwise_agreement"] == 0.5
    lo, hi = wilson_interval(80, 100)
    assert 0 < lo < 0.8 < hi < 1
