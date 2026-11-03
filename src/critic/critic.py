"""Statistical critic: checks the numbers an experiment produced. RULE-BASED, NO LLM.

The experiment script must print ONE JSON object in this format (the Code Executor is told so):

    {"data_origin": "synthetic" | "simulated" | "benchmark" | "real",
     "comparisons": [{"name": "treatment vs baseline", "labels": ["baseline", "treatment"],
                      "a": [raw values for the baseline], "b": [raw values for the treatment],
                      "paired": false,
                      "resampling": {"n_train": 124, "n_test": 54}}]}     (resampling is optional: it marks
                      scores from repeated train/test splits, which get the corrected resampled t-test)

The critic recomputes every statistic itself from the raw values. It never trusts p-values printed
by the script, never executes anything, and never calls a language model.

What the verdict means (and does not):
  SIGNIFICANT           at least one comparison is significant after Holm correction (under BOTH tests, when two apply).
                        This is not proof that the hypothesis is true.
  INCONCLUSIVE          data were analysable but no comparison was significant (or the tests disagreed).
                        This does not show that there is no effect.
  NOT_EVALUABLE         the run succeeded but its output could not be analysed (wrong format, no samples, ...)
  NO_RESULT             the experiment did not run successfully, so there is nothing to check.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.critic.stats import ComparisonResult, StatsInputError, compare_groups, compare_resampled, holm_adjust

MAX_COMPARISONS = 20
MAX_VALUES = 10_000
SMALL_SAMPLE = 10
KNOWN_ORIGINS = {"synthetic", "simulated", "benchmark", "real"}


class Verdict:
    SIGNIFICANT = "SIGNIFICANT"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    NO_RESULT = "NO_RESULT"


@dataclass
class Finding:
    severity: str          # "error" | "warning" | "info"
    code: str
    message: str


@dataclass
class ComparisonReport:
    name: str
    labels: List[str]
    result: Optional[ComparisonResult] = None
    adjusted_primary_p: Optional[float] = None
    adjusted_robust_p: Optional[float] = None
    significant: bool = False
    error: str = ""


@dataclass
class CriticReport:
    verdict: str
    alpha: float
    data_origin: str = "unknown"
    comparisons: List[ComparisonReport] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    summary_lines: List[str] = field(default_factory=list)
    method: str = "rule-based statistics (SciPy); no language model"

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def extract_json(stdout: str) -> Optional[Any]:
    """Return the last line of stdout that is a JSON object, else the whole text if it parses."""
    for line in reversed([ln.strip() for ln in stdout.splitlines() if ln.strip()]):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return None


def _numbers(value: Any) -> Optional[List[float]]:
    if not isinstance(value, list) or len(value) > MAX_VALUES:
        return None
    out: List[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        out.append(float(item))
    return out


class StatisticalCritic:
    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 0.5:
            raise ValueError("alpha must be between 0 and 0.5")
        self.alpha = alpha

    def evaluate(self, stdout: str, execution_status: str = "SUCCESS", output_truncated: bool = False) -> CriticReport:
        report = CriticReport(Verdict.NOT_EVALUABLE, self.alpha)

        if execution_status != "SUCCESS":
            report.verdict = Verdict.NO_RESULT
            report.findings.append(Finding("error", "NO_RESULT",
                f"The experiment ended with {execution_status}, so there are no results to check."))
            return report
        if output_truncated:
            report.findings.append(Finding("error", "OUTPUT_TRUNCATED",
                "The script's output was cut off at the size limit, so it cannot be analysed reliably. "
                "Print far fewer numbers."))
            return report
        data = extract_json(stdout or "")
        if not isinstance(data, dict):
            report.findings.append(Finding("error", "NO_JSON",
                "The script did not print a JSON object in the required format, so there is nothing to analyse."))
            return report
        comps = data.get("comparisons")
        if not isinstance(comps, list) or not comps:
            report.findings.append(Finding("error", "BAD_SCHEMA",
                "The JSON has no non-empty 'comparisons' list with raw values in 'a' and 'b'."))
            return report
        if len(comps) > MAX_COMPARISONS:
            report.findings.append(Finding("error", "TOO_MANY_COMPARISONS",
                f"{len(comps)} comparisons were reported; the limit is {MAX_COMPARISONS}."))
            return report

        self._data_origin_findings(data.get("data_origin"), report)
        self._analyse(comps, report)
        self._decide(report)
        return report

    # ------------------------------------------------------------------ helpers
    def _data_origin_findings(self, origin: Any, report: CriticReport) -> None:
        origin_text = origin.strip().lower() if isinstance(origin, str) else ""
        report.data_origin = origin_text if origin_text in KNOWN_ORIGINS else "unknown"
        if report.data_origin == "unknown":
            report.findings.append(Finding("warning", "DATA_ORIGIN_MISSING",
                "The script did not say where its data came from ('synthetic', 'simulated' or 'real')."))
        elif report.data_origin == "benchmark":
            report.findings.append(Finding("warning", "DATA_BENCHMARK",
                "The data are a small classic benchmark dataset bundled with scikit-learn (iris, wine, breast_cancer or "
                "digits). The result describes that dataset only. It does not show how the methods behave on other data."))
        elif report.data_origin in ("synthetic", "simulated"):
            report.findings.append(Finding("warning", "DATA_SYNTHETIC",
                "The data were generated by the script itself. The statistics describe those simulated numbers "
                "only and say nothing about real systems."))
        else:
            report.findings.append(Finding("warning", "DATA_CLAIMED_REAL",
                "The script claims 'real' data, but the sandbox has no network and no files. Real data could only be "
                "a small dataset bundled with scikit-learn, or numbers typed into the script by the model. "
                "Check which before believing the claim."))

    def _analyse(self, comps: List[Any], report: CriticReport) -> None:
        for index, item in enumerate(comps, start=1):
            name = str(item.get("name", f"comparison {index}"))[:120] if isinstance(item, dict) else f"comparison {index}"
            labels = ["a", "b"]
            if isinstance(item, dict) and isinstance(item.get("labels"), list) and len(item["labels"]) == 2:
                labels = [str(x)[:100] for x in item["labels"]]
            cr = ComparisonReport(name, labels)
            report.comparisons.append(cr)
            if not isinstance(item, dict):
                cr.error = "not an object"
                continue
            a, b = _numbers(item.get("a")), _numbers(item.get("b"))
            if a is None or b is None:
                cr.error = f"'a' and 'b' must be lists of numbers (at most {MAX_VALUES} each)"
                continue
            paired = bool(item.get("paired", False))
            resampling = item.get("resampling")
            try:
                if resampling is not None:
                    n_train = resampling.get("n_train") if isinstance(resampling, dict) else None
                    n_test = resampling.get("n_test") if isinstance(resampling, dict) else None
                    if not paired or not all(isinstance(v, int) and not isinstance(v, bool) for v in (n_train, n_test)):
                        raise StatsInputError("'resampling' needs paired=true and whole numbers n_train and n_test")
                    cr.result = compare_resampled(a, b, n_train, n_test, alpha=self.alpha)
                    report.findings.append(Finding("info", "RESAMPLED_DATA",
                        f"'{name}': the scores come from repeated splits of the same data, so they are not independent. "
                        "The corrected resampled t-test accounts for this."))
                else:
                    cr.result = compare_groups(a, b, paired=paired, alpha=self.alpha)
            except StatsInputError as exc:
                cr.error = str(exc)
                continue
            res = cr.result
            if min(res.n_a, res.n_b) < SMALL_SAMPLE:
                report.findings.append(Finding("warning", "SMALL_SAMPLE",
                    f"'{name}': only {min(res.n_a, res.n_b)} values in the smaller group. Results with fewer than "
                    f"{SMALL_SAMPLE} values per group are unreliable."))
            reported = item.get("reported_p")
            if isinstance(reported, (int, float)) and not isinstance(reported, bool) and math.isfinite(reported):
                if abs(reported - res.primary_p) > 0.01:
                    report.findings.append(Finding("warning", "REPORTED_P_MISMATCH",
                        f"'{name}': the script reported p = {reported:.3g}, but recomputing from its own raw values "
                        f"gives p = {res.primary_p:.3g}. The recomputed value is used."))
        for cr in report.comparisons:
            if cr.error:
                report.findings.append(Finding("error", "COMPARISON_NOT_ANALYSABLE", f"'{cr.name}': {cr.error}."))

    def _decide(self, report: CriticReport) -> None:
        valid = [c for c in report.comparisons if c.result is not None]
        if not valid:
            report.verdict = Verdict.NOT_EVALUABLE
            return
        primary = holm_adjust([c.result.primary_p for c in valid])
        with_robust = [c for c in valid if c.result.robust_p is not None]
        robust_adjusted = dict(zip(map(id, with_robust), holm_adjust([c.result.robust_p for c in with_robust])))
        robust = [robust_adjusted.get(id(c)) for c in valid]
        if len(valid) > 1:
            report.findings.append(Finding("info", "MULTIPLE_COMPARISONS",
                f"{len(valid)} comparisons were tested, so Holm correction was applied to the p-values."))
        for c, p_adj, r_adj in zip(valid, primary, robust):
            c.adjusted_primary_p, c.adjusted_robust_p = p_adj, r_adj
            c.significant = p_adj < self.alpha and (r_adj is None or r_adj < self.alpha)
            if r_adj is not None and (p_adj < self.alpha) != (r_adj < self.alpha):
                report.findings.append(Finding("warning", "TESTS_DISAGREE",
                    f"'{c.name}': {c.result.primary_test} and {c.result.robust_test} give different answers. "
                    "The result is not robust."))
            res = c.result
            level = int(round(res.ci_level * 100))
            second = (f"; {res.robust_test} p = {_fmt_p(res.robust_p)} (adjusted {_fmt_p(r_adj)})"
                      if res.robust_p is not None else "")
            report.summary_lines.append(
                f"{c.name}: {c.labels[1]} minus {c.labels[0]} = {res.mean_diff:.3g} "
                f"({level}% CI {res.ci_low:.3g} to {res.ci_high:.3g}); {res.primary_test} p = {_fmt_p(res.primary_p)} "
                f"(adjusted {_fmt_p(p_adj)}){second}; {res.effect_size_name} = {res.effect_size:.2f}; "
                f"n = {res.n_a} and {res.n_b}."
            )
        if any(c.significant for c in valid):
            report.verdict = Verdict.SIGNIFICANT
            report.findings.append(Finding("info", "SIGNIFICANT_NOT_PROOF",
                "A significant result means the data would be unlikely if there were no difference. It does not "
                "prove the hypothesis, and it is only as good as the experiment's design."))
        else:
            report.verdict = Verdict.INCONCLUSIVE
            report.findings.append(Finding("info", "NOT_SIGNIFICANT_NOT_NO_EFFECT",
                "A result that is not significant does not show that there is no effect. The sample may be too "
                "small to detect one."))
