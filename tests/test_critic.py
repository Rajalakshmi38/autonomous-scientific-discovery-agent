"""Tests for the statistical critic. All numbers here are SYNTHETIC test data.

Reference values were worked out by hand or from textbook tables, not just copied from the code:
  * Welch t-test on [1..5] vs [2..6]: t = 1.0, df = 8, p = 0.3466, 95% CI -1.306..3.306 (b minus a = +1)
  * Holm adjustment of [0.01, 0.04, 0.03] is [0.03, 0.06, 0.06]
  * paired example: differences [1,2,2,3,4] -> t = 4.707 on 4 df, p = 0.0093
"""
import json

import pytest

from config.settings import ConfigError, load_settings
from src.critic.critic import MAX_VALUES, StatisticalCritic, Verdict, extract_json
from src.critic.stats import StatsInputError, compare_groups, holm_adjust


def payload(a, b, origin="synthetic", paired=False, **extra):
    comp = {"name": "treatment vs baseline", "labels": ["baseline", "treatment"], "a": a, "b": b, "paired": paired, **extra}
    return json.dumps({"data_origin": origin, "comparisons": [comp]})


BASE = [1.0, 1.1, 0.9, 1.2, 0.8, 1.05, 0.95, 1.15, 0.85, 1.0, 1.02, 0.98]
SHIFTED = [x + 2.0 for x in BASE]                                   # a huge, obvious difference
SIMILAR = [x + 0.01 * (i % 3 - 1) for i, x in enumerate(BASE)]      # practically identical


def codes(report):
    return {f.code for f in report.findings}


# ---------------------------------------------------------------- statistics against known values
def test_welch_test_matches_hand_calculated_values():
    r = compare_groups([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    assert r.mean_diff == pytest.approx(1.0)
    assert r.primary_statistic == pytest.approx(1.0, abs=1e-6)
    assert r.primary_p == pytest.approx(0.3466, abs=1e-3)
    assert (r.ci_low, r.ci_high) == (pytest.approx(-1.306, abs=2e-3), pytest.approx(3.306, abs=2e-3))
    assert r.effect_size == pytest.approx(0.571, abs=2e-3)         # Hedges' g = 0.632 * (1 - 3/31)
    assert r.primary_test == "Welch t-test" and r.robust_test == "Mann-Whitney U"


def test_direction_is_treatment_minus_baseline():
    assert compare_groups([2, 3, 4, 5, 6], [1, 2, 3, 4, 5]).mean_diff == pytest.approx(-1.0)


def test_paired_test_matches_hand_calculation():
    r = compare_groups([1, 2, 3, 4, 5], [2, 4, 5, 7, 9], paired=True)
    assert r.paired and r.mean_diff == pytest.approx(2.4)
    assert r.primary_statistic == pytest.approx(4.707, abs=2e-3)
    assert r.primary_p == pytest.approx(0.0093, abs=1e-3)
    assert r.robust_p == pytest.approx(0.0625, abs=1e-4)           # with 5 pairs Wilcoxon cannot go below 0.0625


def test_holm_adjustment_matches_known_values():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5]) == [0.5]
    assert holm_adjust([0.9, 0.8]) == pytest.approx([1.0, 1.0])    # 2 x 0.8 is capped at 1, and adjusted values never decrease


@pytest.mark.parametrize("a,b,kwargs", [
    ([1], [1, 2, 3], {}),                                    # too few values
    ([1, 2, float("nan")], [1, 2, 3], {}),                   # not a number
    ([1, 2, float("inf")], [1, 2, 3], {}),
    ([3, 3, 3], [5, 5, 5], {}),                              # no variation in either group
    ([1, 2, 3], [1, 2], {"paired": True}),                   # paired needs equal sizes
    ([1, 2, 3], [2, 3, 4], {"paired": True}),                # identical differences
])
def test_unanalysable_data_is_rejected(a, b, kwargs):
    with pytest.raises(StatsInputError):
        compare_groups(a, b, **kwargs)


@pytest.mark.filterwarnings("ignore:Precision loss")
def test_one_constant_group_is_still_analysable():
    r = compare_groups([3.0, 3.0, 3.0, 3.0], [1.0, 2.0, 3.0, 4.0])      # Welch's test copes with one group having no spread
    assert 0 < r.primary_p <= 1 and r.mean_diff == pytest.approx(-0.5)


def test_invalid_alpha_is_rejected():
    with pytest.raises(ValueError):
        compare_groups([1, 2], [3, 4], alpha=0.7)
    with pytest.raises(ValueError):
        StatisticalCritic(alpha=0)


# ---------------------------------------------------------------- verdicts
def test_failed_execution_gives_no_result():
    r = StatisticalCritic().evaluate("", execution_status="FAILED")
    assert r.verdict == Verdict.NO_RESULT and "NO_RESULT" in codes(r)


@pytest.mark.parametrize("stdout,code", [
    ("just some text, no json", "NO_JSON"),
    ("[1, 2, 3]", "NO_JSON"),
    (json.dumps({"data_origin": "synthetic"}), "BAD_SCHEMA"),
    (json.dumps({"comparisons": []}), "BAD_SCHEMA"),
])
def test_unusable_output_is_not_evaluable(stdout, code):
    r = StatisticalCritic().evaluate(stdout)
    assert r.verdict == Verdict.NOT_EVALUABLE and code in codes(r)


def test_truncated_output_is_not_trusted():
    r = StatisticalCritic().evaluate(payload(BASE, SHIFTED), output_truncated=True)
    assert r.verdict == Verdict.NOT_EVALUABLE and "OUTPUT_TRUNCATED" in codes(r)


@pytest.mark.parametrize("bad", [
    {"a": "not a list", "b": [1, 2, 3]},
    {"a": [1, 2, "x"], "b": [1, 2, 3]},
    {"a": [1, True, 3], "b": [1, 2, 3]},                     # booleans are not measurements
    {"a": list(range(MAX_VALUES + 1)), "b": [1, 2, 3]},      # absurdly large list
    {"a": [1.0], "b": [2.0, 3.0]},
])
def test_bad_comparison_data_is_not_evaluable(bad):
    stdout = json.dumps({"data_origin": "synthetic", "comparisons": [bad]})
    r = StatisticalCritic().evaluate(stdout)
    assert r.verdict == Verdict.NOT_EVALUABLE and "COMPARISON_NOT_ANALYSABLE" in codes(r)


def test_large_clear_difference_is_significant_but_never_called_proof():
    r = StatisticalCritic().evaluate(payload(BASE, SHIFTED))
    assert r.verdict == Verdict.SIGNIFICANT
    assert r.comparisons[0].significant and r.comparisons[0].adjusted_primary_p < 0.001
    assert {"DATA_SYNTHETIC", "SIGNIFICANT_NOT_PROOF"} <= codes(r)
    assert "treatment minus baseline" in r.summary_lines[0] and "CI" in r.summary_lines[0]


def test_tiny_difference_is_inconclusive_and_says_so_carefully():
    r = StatisticalCritic().evaluate(payload(BASE, SIMILAR))
    assert r.verdict == Verdict.INCONCLUSIVE and "NOT_SIGNIFICANT_NOT_NO_EFFECT" in codes(r)


def test_small_samples_are_flagged():
    r = StatisticalCritic().evaluate(payload([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]))
    assert "SMALL_SAMPLE" in codes(r)


def test_scripts_own_p_value_is_not_trusted():
    r = StatisticalCritic().evaluate(payload([1, 2, 3, 4, 5], [2, 3, 4, 5, 6], reported_p=0.001))
    assert "REPORTED_P_MISMATCH" in codes(r)
    assert r.comparisons[0].result.primary_p == pytest.approx(0.3466, abs=1e-3)   # recomputed, not 0.001


def test_disagreeing_tests_prevent_a_significant_verdict():
    r = StatisticalCritic().evaluate(payload([1, 2, 3, 4, 5], [2, 4, 5, 7, 9], paired=True))
    assert r.comparisons[0].result.primary_p < 0.05 <= r.comparisons[0].result.robust_p
    assert r.verdict == Verdict.INCONCLUSIVE and "TESTS_DISAGREE" in codes(r)


def test_multiple_comparisons_are_corrected():
    comps = [{"name": f"c{i}", "a": BASE, "b": [x + 0.3 for x in BASE]} for i in range(3)]
    r = StatisticalCritic().evaluate(json.dumps({"data_origin": "synthetic", "comparisons": comps}))
    assert "MULTIPLE_COMPARISONS" in codes(r)
    c = r.comparisons[0]
    assert c.adjusted_primary_p >= c.result.primary_p          # correction can only make p larger


def test_one_bad_comparison_does_not_hide_a_good_one():
    comps = [{"name": "bad", "a": [1], "b": [2]}, {"name": "good", "a": BASE, "b": SHIFTED}]
    r = StatisticalCritic().evaluate(json.dumps({"data_origin": "synthetic", "comparisons": comps}))
    assert r.verdict == Verdict.SIGNIFICANT and "COMPARISON_NOT_ANALYSABLE" in codes(r)


@pytest.mark.parametrize("origin,code", [("real", "DATA_CLAIMED_REAL"), (None, "DATA_ORIGIN_MISSING"), ("made up", "DATA_ORIGIN_MISSING")])
def test_data_origin_is_always_questioned(origin, code):
    stdout = json.dumps({"data_origin": origin, "comparisons": [{"a": BASE, "b": SHIFTED}]})
    r = StatisticalCritic().evaluate(stdout)
    assert code in codes(r)


def test_alpha_changes_the_decision():
    stdout = payload(BASE, [x + 0.15 for x in BASE])           # Welch p = 0.0048, Mann-Whitney p = 0.0100
    assert StatisticalCritic(0.05).evaluate(stdout).verdict == Verdict.SIGNIFICANT
    assert StatisticalCritic(0.001).evaluate(stdout).verdict == Verdict.INCONCLUSIVE     # stricter threshold, same data


def test_json_is_found_in_noisy_output_and_report_is_serialisable():
    noisy = "debug line\n{not json\n" + payload(BASE, SHIFTED) + "\n"
    assert extract_json(noisy)["data_origin"] == "synthetic"
    report = StatisticalCritic().evaluate(noisy)
    assert json.loads(json.dumps(report.to_dict()))["verdict"] == Verdict.SIGNIFICANT


def test_critic_setting_is_validated():
    assert load_settings(env={}, load_env_file=False).critic_alpha == 0.05
    with pytest.raises(ConfigError):
        load_settings(env={"CRITIC_ALPHA": "0.9"}, load_env_file=False)


# ---------------------------------------------------------------- repeated train/test splits (Nadeau-Bengio)
from src.critic.stats import compare_resampled


def test_resampled_test_matches_hand_calculation():
    # differences [.1,.2,.3,0,.4]: mean .2, variance .025, k = 5, n_test/n_train = .5
    # corrected se = sqrt((1/5 + .5) * .025) = .13229 -> t = 1.5119 on 4 df -> p = 0.205
    r = compare_resampled([0, 0, 0, 0, 0], [0.1, 0.2, 0.3, 0.0, 0.4], n_train=100, n_test=50)
    assert r.primary_statistic == pytest.approx(1.5119, abs=1e-3)
    assert r.primary_p == pytest.approx(0.2051, abs=1e-3)
    assert (r.ci_low, r.ci_high) == (pytest.approx(-0.167, abs=2e-3), pytest.approx(0.567, abs=2e-3))
    assert r.robust_p is None and "Nadeau" in r.primary_test


def test_the_correction_makes_resampled_results_less_optimistic():
    a, b = [0, 0, 0, 0, 0], [0.1, 0.2, 0.3, 0.0, 0.4]
    assert compare_resampled(a, b, 100, 50).primary_p > compare_groups(a, b, paired=True).primary_p


def test_resampled_edge_cases():
    same = compare_resampled([0.9] * 6, [0.9] * 6, 100, 50)
    assert same.primary_p == 1.0 and same.mean_diff == 0
    for bad in (lambda: compare_resampled([0.5] * 4, [0.6] * 4, 100, 50),          # identical non-zero differences
                lambda: compare_resampled([1, 2, 3], [1, 2], 100, 50),
                lambda: compare_resampled([1, 2, 3], [2, 4, 5], 0, 50)):
        with pytest.raises(StatsInputError):
            bad()


A_SCORES = [0.70, 0.72, 0.69, 0.71, 0.73, 0.70, 0.68, 0.72, 0.71, 0.70]
B_SCORES = [0.90, 0.93, 0.88, 0.92, 0.95, 0.91, 0.87, 0.94, 0.90, 0.91]


def resampled_payload(a, b, origin="benchmark", resampling=None, paired=True):
    comp = {"name": "B vs A", "labels": ["A", "B"], "a": a, "b": b, "paired": paired,
            "resampling": {"n_train": 124, "n_test": 54} if resampling is None else resampling}
    return json.dumps({"data_origin": origin, "comparisons": [comp]})


def test_critic_uses_the_corrected_test_for_resampled_scores():
    r = StatisticalCritic().evaluate(resampled_payload(A_SCORES, B_SCORES))
    assert r.verdict == Verdict.SIGNIFICANT
    assert r.comparisons[0].result.primary_test.startswith("corrected resampled")
    assert {"RESAMPLED_DATA", "DATA_BENCHMARK"} <= codes(r)
    assert "TESTS_DISAGREE" not in codes(r) and "Mann-Whitney" not in r.summary_lines[0]


def test_resampled_scores_with_no_real_difference_are_inconclusive():
    r = StatisticalCritic().evaluate(resampled_payload(A_SCORES, [x + (0.02 if i % 2 else -0.02) for i, x in enumerate(A_SCORES)]))
    assert r.verdict == Verdict.INCONCLUSIVE


@pytest.mark.parametrize("resampling,paired", [
    ({"n_train": 124, "n_test": 54}, False),            # resampling only makes sense for paired scores
    ({"n_train": "124", "n_test": 54}, True),
    ({"n_train": 124}, True),
    ("lots", True),
    ({"n_train": True, "n_test": 54}, True),
])
def test_bad_resampling_information_is_rejected(resampling, paired):
    r = StatisticalCritic().evaluate(resampled_payload(A_SCORES, B_SCORES, resampling=resampling, paired=paired))
    assert r.verdict == Verdict.NOT_EVALUABLE and "COMPARISON_NOT_ANALYSABLE" in codes(r)


def test_long_labels_and_names_are_not_cut_off():
    label_a, label_b = "random_forest(max_depth=None, n_estimators=50)", "logistic_regression(C=1) + scaling"
    stdout = json.dumps({"data_origin": "benchmark", "comparisons": [{
        "name": "iris (accuracy)", "labels": [label_a, label_b], "a": A_SCORES, "b": B_SCORES, "paired": True,
        "resampling": {"n_train": 105, "n_test": 45}}]})
    line = StatisticalCritic().evaluate(stdout).summary_lines[0]
    assert label_a in line and label_b in line and line.startswith("iris (accuracy):")
