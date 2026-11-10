"""Experiment specifications: the model DESIGNS an experiment, vetted code WRITES the script.

Why: small local models (1-3B parameters) write unreliable Python (missing imports, wrong APIs, cut-off
code). Here the model only fills in a small form (which dataset, which two methods, which metric). The form
is validated against fixed lists, and the Python script is rendered from a template that was written and
tested by hand, so the script is correct by construction.

Honest limits:
  * Only the experiments below can be run. They compare two methods, or two settings of one method, on a
    small benchmark dataset bundled with scikit-learn (no internet is needed).
  * The link between the hypothesis and the chosen experiment is made by the model and is NOT verified.
  * Nothing from the model is ever pasted into the script as code: only whitelisted names and numbers.
"""
from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

DATASETS = ("iris", "wine", "breast_cancer", "digits")
METRICS = ("accuracy", "f1_macro")
MIN_RESAMPLES, MAX_RESAMPLES = 10, 30
TEST_SIZE = 0.3

# model name -> {parameter name: allowed values}. JSON null means Python None.
MODEL_OPTIONS: Dict[str, Dict[str, list]] = {
    "logistic_regression": {"C": [0.01, 0.1, 1, 10, 100]},
    "decision_tree": {"max_depth": [1, 2, 3, 5, 10, None]},
    "random_forest": {"n_estimators": [5, 10, 50, 100], "max_depth": [2, 5, None]},
    "knn": {"n_neighbors": [1, 3, 5, 11, 25]},
    "naive_bayes": {},
    "svm": {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10]},
}


class SpecError(ValueError):
    """The experiment form is invalid. The message says what is allowed, so it can be shown to the model."""


@dataclass
class ModelSpec:
    model: str
    params: Dict[str, Any] = field(default_factory=dict)
    scale: bool = False

    def describe(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        text = f"{self.model}({args})" if args else self.model
        return text + " + scaling" if self.scale else text


@dataclass
class ExperimentSpec:
    dataset: str
    baseline: ModelSpec
    treatment: ModelSpec
    metric: str = "accuracy"
    n_resamples: int = 20
    test_size: float = TEST_SIZE
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset, "metric": self.metric, "n_resamples": self.n_resamples,
            "test_size": self.test_size, "seed": self.seed,
            "baseline": {"model": self.baseline.model, "params": dict(self.baseline.params), "scale": self.baseline.scale},
            "treatment": {"model": self.treatment.model, "params": dict(self.treatment.params), "scale": self.treatment.scale},
        }

    def summary(self) -> str:
        return (f"{self.dataset}: {self.treatment.describe()} vs {self.baseline.describe()}, "
                f"{self.metric}, {self.n_resamples} resamples")


def options_text() -> str:
    """The allowed choices, written for the prompt. Generated from the same tables the validator uses."""
    lines = ["datasets: " + ", ".join(DATASETS), "metrics: " + ", ".join(METRICS),
             f"n_resamples: a whole number from {MIN_RESAMPLES} to {MAX_RESAMPLES}", "models and their allowed parameters:"]
    for name, params in MODEL_OPTIONS.items():
        shown = "; ".join(f"{p} in {json.dumps(v)}" for p, v in params.items()) or "no parameters"
        lines.append(f"  - {name}: {shown}")
    lines.append('"scale": true standardises the features first (allowed for every model).')
    return "\n".join(lines)


def _canonical(value: Any, allowed: list, label: str) -> Any:
    for candidate in allowed:
        if isinstance(value, bool) or isinstance(candidate, bool):
            continue
        if value is None and candidate is None:
            return None
        if value is not None and candidate is not None and not isinstance(value, (list, dict)) and value == candidate:
            return candidate                      # always use OUR value, never the model's object
    raise SpecError(f"{label} = {json.dumps(value)} is not allowed. Allowed values: {json.dumps(allowed)}")


def _validate_model(raw: Any, label: str) -> ModelSpec:
    if not isinstance(raw, dict):
        raise SpecError(f"'{label}' must be an object like {{\"model\": \"knn\", \"params\": {{}}, \"scale\": false}}")
    name = raw.get("model")
    if name not in MODEL_OPTIONS:
        raise SpecError(f"{label}.model = {json.dumps(name)} is not allowed. Allowed models: {sorted(MODEL_OPTIONS)}")
    raw_params = raw.get("params", {})
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, dict):
        raise SpecError(f"{label}.params must be an object")
    params: Dict[str, Any] = {}
    for key, value in raw_params.items():
        if key not in MODEL_OPTIONS[name]:
            raise SpecError(f"{label}.params has '{key}', which {name} does not allow. Allowed: {sorted(MODEL_OPTIONS[name]) or 'none'}")
        params[key] = _canonical(value, MODEL_OPTIONS[name][key], f"{label}.params.{key}")
    scale = raw.get("scale", False)
    if not isinstance(scale, bool):
        raise SpecError(f"{label}.scale must be true or false")
    return ModelSpec(name, params, scale)


def validate_spec(raw: Any) -> ExperimentSpec:
    """Check a decoded JSON object. Raises SpecError with a message the model can act on."""
    if not isinstance(raw, dict):
        raise SpecError("The experiment must be one JSON object.")
    dataset = raw.get("dataset")
    if dataset not in DATASETS:
        raise SpecError(f"dataset = {json.dumps(dataset)} is not allowed. Allowed datasets: {list(DATASETS)}")
    metric = raw.get("metric", "accuracy")
    if metric not in METRICS:
        raise SpecError(f"metric = {json.dumps(metric)} is not allowed. Allowed metrics: {list(METRICS)}")
    n = raw.get("n_resamples", 20)
    if isinstance(n, bool) or not isinstance(n, (int, float)) or n != int(n) or not MIN_RESAMPLES <= int(n) <= MAX_RESAMPLES:
        raise SpecError(f"n_resamples must be a whole number from {MIN_RESAMPLES} to {MAX_RESAMPLES}")
    baseline, treatment = _validate_model(raw.get("baseline"), "baseline"), _validate_model(raw.get("treatment"), "treatment")
    if baseline == treatment:
        raise SpecError("baseline and treatment are identical, so there is nothing to compare. Change the model, a parameter or scale.")
    return ExperimentSpec(dataset, baseline, treatment, metric, int(n))


def parse_spec_text(text: str) -> Dict[str, Any]:
    """Find the first JSON object in a model reply."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text or ""):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise SpecError("No JSON object was found in the reply. Reply with exactly one JSON object.")


def default_spec() -> ExperimentSpec:
    """A generic experiment, used only when the model could not produce a valid one. It does NOT test the hypothesis."""
    return ExperimentSpec("iris", ModelSpec("logistic_regression", {"C": 1}), ModelSpec("random_forest", {"n_estimators": 50}))


# --------------------------------------------------------------------------- tolerant reading of the model's form
# Small models make small slips ("Random Forest", n_neighbors = 7). Instead of rejecting those, they are mapped
# to the nearest allowed option and every change is REPORTED. The script still only ever uses whitelisted values.
_MODEL_WORDS = {
    "logistic_regression": ("logisticregression", "logreg"),
    "decision_tree": ("decisiontree",),
    "random_forest": ("randomforest",),
    "knn": ("knn", "knearest", "nearestneighbor", "nearestneighbour", "kneighbors"),
    "naive_bayes": ("naivebayes", "gaussiannb"),
    "svm": ("svm", "svc", "supportvector"),
}
_DATASET_WORDS = {"iris": ("iris",), "wine": ("wine",), "breast_cancer": ("breastcancer",), "digits": ("digits", "handwrittendigit")}
_METRIC_WORDS = {"accuracy": ("accuracy", "acc"), "f1_macro": ("f1",)}
_PARAM_WORDS = {"k": "n_neighbors", "neighbors": "n_neighbors", "neighbours": "n_neighbors", "n_neighbours": "n_neighbors",
                "trees": "n_estimators", "n_trees": "n_estimators", "depth": "max_depth", "c": "C"}


def _squash(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _lookup(words: Dict[str, tuple], text: Any) -> Optional[str]:
    squashed = _squash(text)
    for canonical, aliases in words.items():
        if squashed == _squash(canonical) or any(a in squashed for a in aliases):
            return canonical
    return None


def _snap(value: Any, allowed: list) -> Any:
    """The nearest allowed value for a number; strings are matched ignoring case. Returns the value unchanged if no match."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in ("none", "null") and None in allowed:
            return None
        try:                                           # models often write numbers as text, like "50"
            value = float(text)
        except ValueError:
            value = value
    if isinstance(value, (int, float)) and any(isinstance(a, (int, float)) and not isinstance(a, bool) for a in allowed):
        numbers = [a for a in allowed if isinstance(a, (int, float)) and not isinstance(a, bool)]
        return min(numbers, key=lambda a: (abs(a - value), a))
    if isinstance(value, str):
        for a in allowed:
            if isinstance(a, str) and a.lower() == value.strip().lower():
                return a
    return value


def normalise_raw(raw: Any) -> Tuple[Any, List[str]]:
    """Map a model's slightly-off form onto the allowed options. Returns (cleaned form, notes about every change)."""
    if not isinstance(raw, dict):
        return raw, []
    notes: List[str] = []
    out = dict(raw)

    for key, words, allowed in (("dataset", _DATASET_WORDS, DATASETS), ("metric", _METRIC_WORDS, METRICS)):
        if key in out and out[key] not in allowed and isinstance(out[key], str):
            found = _lookup(words, out[key])
            if found:
                notes.append(f"{key} '{out[key]}' -> '{found}'")
                out[key] = found
    n = out.get("n_resamples", 20)
    if isinstance(n, (int, float)) and not isinstance(n, bool):
        clamped = min(MAX_RESAMPLES, max(MIN_RESAMPLES, int(round(n))))
        if clamped != n:
            notes.append(f"n_resamples {n} -> {clamped}")
        out["n_resamples"] = clamped
    else:
        out["n_resamples"] = 20

    for side in ("baseline", "treatment"):
        item = out.get(side)
        if not isinstance(item, dict):
            continue
        item = dict(item)
        name = item.get("model")
        if isinstance(name, str) and name not in MODEL_OPTIONS:
            found = _lookup(_MODEL_WORDS, name)
            if found:
                notes.append(f"{side}.model '{name}' -> '{found}'")
                item["model"] = found
        options = MODEL_OPTIONS.get(item.get("model"), None)
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        for alias in ("scale", "scaling", "standardize", "standardise"):      # a scaling flag written inside params
            if alias in params and (options is None or alias not in options):
                if "scale" not in item and isinstance(params[alias], (bool, int, str)):
                    item["scale"] = params[alias]
                    notes.append(f"{side}.params.{alias} moved to {side}.scale")
                else:
                    notes.append(f"{side}.params.{alias} ignored (feature scaling is set with {side}.scale)")
                params = {k: v for k, v in params.items() if k != alias}
        cleaned: Dict[str, Any] = {}
        for key, value in params.items():
            target = key if (options is not None and key in options) else _PARAM_WORDS.get(str(key).lower(), key)
            if options is None or target not in options:
                notes.append(f"{side}.params.{key} ignored (not supported for {item.get('model')})")
                continue
            if target != key:
                notes.append(f"{side}.params.{key} -> {target}")
            snapped = _snap(value, options[target])
            if snapped != value and not (isinstance(value, float) and value == snapped):
                notes.append(f"{side}.params.{target} {json.dumps(value)} -> {json.dumps(snapped)}")
            cleaned[target] = snapped
        item["params"] = cleaned
        scale = item.get("scale", False)
        if isinstance(scale, str):
            item["scale"] = scale.strip().lower() in ("true", "yes", "1")
            notes.append(f"{side}.scale '{scale}' -> {item['scale']}")
        elif isinstance(scale, (int, float)) and not isinstance(scale, bool):
            item["scale"] = bool(scale)
        out[side] = item
    return out, notes


def confounding_factors(spec: "ExperimentSpec") -> List[str]:
    """What differs between baseline and treatment. A fair experiment changes exactly ONE thing."""
    a, b = spec.baseline, spec.treatment
    differing: List[str] = []
    if a.model != b.model:
        differing.append("the model")                    # different models have different parameters by nature
    else:
        differing += [f"the parameter {k}" for k in sorted(set(a.params) | set(b.params)) if a.params.get(k) != b.params.get(k)]
    if a.scale != b.scale:
        differing.append("feature scaling")
    return differing


def validate_spec_tolerant(raw: Any, single_factor: bool = True) -> Tuple["ExperimentSpec", List[str]]:
    """Like validate_spec, but first maps small slips onto allowed options. Returns (spec, notes about changes).

    With single_factor=True (the default) a design that changes more than one thing at once is rejected,
    because any difference could then be caused by either change.
    """
    cleaned, notes = normalise_raw(raw)
    spec = validate_spec(cleaned)
    factors = confounding_factors(spec)
    if single_factor and len(factors) != 1:
        raise SpecError("The baseline and treatment differ in " + (" and ".join(factors) or "nothing") +
                        ". Change exactly ONE thing: the model, OR one parameter of the same model, OR feature scaling.")
    return spec, notes


# --------------------------------------------------------------------------- rule-based design from the hypothesis text
_FLAGS = re.IGNORECASE
_TEXT_MODELS = {                              # whole-word patterns, so ordinary words are never mistaken for a model
    "logistic_regression": re.compile(r"\blogistic[ -]regression\b|\blogreg\b", _FLAGS),
    "decision_tree": re.compile(r"\bdecision[ -]trees?\b", _FLAGS),
    "random_forest": re.compile(r"\brandom[ -]forests?\b", _FLAGS),
    "knn": re.compile(r"\bk[ -]?nn\b|\bk[ -]nearest\b|\bnearest[ -]neighbou?rs?\b", _FLAGS),
    "naive_bayes": re.compile(r"\bna[iï]ve[ -]bayes\b", _FLAGS),
    "svm": re.compile(r"\bsvms?\b|\bsvc\b|\bsupport[ -]vector\b", _FLAGS),
}
_TEXT_DATASETS = {
    "iris": re.compile(r"\biris\b", _FLAGS), "wine": re.compile(r"\bwine\b", _FLAGS),
    "breast_cancer": re.compile(r"\bbreast[ -_]cancer\b", _FLAGS), "digits": re.compile(r"\bdigits?\b", _FLAGS),
}
_SCALING_WORDS = re.compile(r"scal(?:e|ed|ing)|standardi[sz]|normali[sz]", re.IGNORECASE)
_ONE_MODEL_SETTINGS = {                       # model -> (low setting, high setting) of its main knob
    "random_forest": ({"n_estimators": 10}, {"n_estimators": 100}),
    "knn": ({"n_neighbors": 1}, {"n_neighbors": 11}),
    "decision_tree": ({"max_depth": 2}, {"max_depth": None}),
    "logistic_regression": ({"C": 0.1}, {"C": 10}),
    "svm": ({"C": 0.1}, {"C": 10}),
}


_CITATION = re.compile(r"\[[^\]]*\]")
_SETTING_WORDS = {                                # model -> (words that name its main setting, parameter name)
    "random_forest": (re.compile(r"\btrees?\b|\bestimators?\b|n_estimators", re.IGNORECASE), "n_estimators"),
    "knn": (re.compile(r"\bneighbou?rs?\b|\bk\s*=|\bvalue of k\b|\bchoice of k\b", re.IGNORECASE), "n_neighbors"),
}
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3})(?![\w.%])")


def _nearest_allowed(value: int, allowed: list) -> int:
    """The allowed value nearest to ``value`` (the lower one on a tie)."""
    return min((a for a in allowed if isinstance(a, int) and not isinstance(a, bool)), key=lambda a: (abs(a - value), a))


def _setting_pair(model: str, text: str) -> Optional[Tuple[str, int, int]]:
    """(parameter, low, high) when the text compares two values of a model's main setting, such as
    'the number of trees ... from 10 to 100'. Only numbers close to the setting's name are used, and they are
    moved to the nearest allowed value. Returns None when fewer than two different values are found."""
    if model not in _SETTING_WORDS:
        return None
    words, param = _SETTING_WORDS[model]
    clean = _CITATION.sub(" ", text or "")
    hit = words.search(clean)
    if not hit:
        return None
    window = clean[max(0, hit.start() - 30): hit.end() + 90]
    allowed = MODEL_OPTIONS[model][param]
    values = sorted({_nearest_allowed(int(n), allowed) for n in _NUMBER.findall(window)})
    if len(values) < 2:
        return None
    return param, values[0], values[-1]


def design_mismatch_note(spec: "ExperimentSpec", statement: str) -> str:
    """Plain check, no language model: does the design contain what the hypothesis names? Returns '' when it does."""
    clean = _CITATION.sub(" ", statement or "")
    named = [name for name, pattern in _TEXT_MODELS.items() if pattern.search(clean)]
    used = {spec.baseline.model, spec.treatment.model}
    notes: List[str] = []
    if named:
        missing = [m for m in named if m not in used]
        extra = sorted(used - set(named))
        if missing:
            notes.append("the hypothesis names " + ", ".join(missing) + ", which the experiment does not use")
        if extra:
            notes.append("the experiment uses " + ", ".join(extra) + ", which the hypothesis does not mention")
    else:
        notes.append("the hypothesis names none of the methods the experiment can run (" + ", ".join(sorted(_TEXT_MODELS))
                     + "), so the experiment is only a stand-in for the topic")
    arms_differ_in_scaling = spec.baseline.scale != spec.treatment.scale
    if arms_differ_in_scaling and not _SCALING_WORDS.search(clean):
        notes.append("the experiment compares feature scaling, but the hypothesis does not mention scaling")
    if _SCALING_WORDS.search(clean) and not arms_differ_in_scaling:
        notes.append("the hypothesis is about feature scaling, but the two arms of the experiment do not differ in scaling")
    return ("Possible mismatch with the hypothesis: " + "; ".join(notes) + ".") if notes else ""


def spec_from_text(text: str) -> Tuple[Optional["ExperimentSpec"], str]:
    """Build a design from keywords in a hypothesis by fixed rules. NOT a language model. Returns (spec or None, explanation)."""
    text = text or ""
    found = sorted(((m.start(), name) for name, pattern in _TEXT_MODELS.items() for m in [pattern.search(text)] if m),
                   key=lambda pair: pair[0])
    models = [name for _, name in found]               # in order of first mention
    dataset = next((name for name, pattern in _TEXT_DATASETS.items() if pattern.search(text)), None)
    scaling = bool(_SCALING_WORDS.search(text))

    for model in models:                                 # two values of one setting, e.g. "trees from 10 to 100"
        pair = _setting_pair(model, text)
        if pair:
            param, low, high = pair
            spec = ExperimentSpec(dataset or "iris", ModelSpec(model, {param: low}, False), ModelSpec(model, {param: high}, False))
            return spec, f"the hypothesis compares two values of {param} for {model} ({low} and {high}), so they were compared"
    if scaling:
        model = models[0] if models else "knn"
        spec = ExperimentSpec(dataset or "wine", ModelSpec(model, {}, False), ModelSpec(model, {}, True))
        return spec, f"the hypothesis mentions scaling, so '{model}' was compared with and without feature scaling"
    if len(models) >= 2:
        spec = ExperimentSpec(dataset or "iris", ModelSpec(models[1], {}, False), ModelSpec(models[0], {}, False))
        return spec, f"the hypothesis mentions {models[0]} and {models[1]}, so they were compared"
    if len(models) == 1 and models[0] in _ONE_MODEL_SETTINGS:
        low, high = _ONE_MODEL_SETTINGS[models[0]]
        spec = ExperimentSpec(dataset or "iris", ModelSpec(models[0], dict(low), False), ModelSpec(models[0], dict(high), False))
        return spec, f"the hypothesis mentions only {models[0]}, so two settings of it were compared"
    return None, "no machine-learning method or scaling was found in the hypothesis"


_TEMPLATE = '''\
# Rendered from a validated experiment specification (src/experiments/spec.py). Not written by the model.
import json, math, warnings
warnings.filterwarnings("ignore")
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SPEC = __SPEC__
LABELS = __LABELS__
LOADERS = {"iris": datasets.load_iris, "wine": datasets.load_wine,
           "breast_cancer": datasets.load_breast_cancer, "digits": datasets.load_digits}

def build(m):
    p, seed = dict(m["params"]), SPEC["seed"]
    if m["model"] == "logistic_regression":
        est = LogisticRegression(max_iter=1000, **p)
    elif m["model"] == "decision_tree":
        est = DecisionTreeClassifier(random_state=seed, **p)
    elif m["model"] == "random_forest":
        est = RandomForestClassifier(random_state=seed, **p)
    elif m["model"] == "knn":
        est = KNeighborsClassifier(**p)
    elif m["model"] == "naive_bayes":
        est = GaussianNB()
    else:
        est = SVC(**p)
    return make_pipeline(StandardScaler(), est) if m["scale"] else est

def score(y_true, y_pred):
    if SPEC["metric"] == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro"))
    return float(accuracy_score(y_true, y_pred))

X, y = LOADERS[SPEC["dataset"]](return_X_y=True)
splitter = StratifiedShuffleSplit(n_splits=SPEC["n_resamples"], test_size=SPEC["test_size"], random_state=SPEC["seed"])
a, b, n_test = [], [], 0
for train, test in splitter.split(X, y):
    n_test = len(test)
    a.append(score(y[test], build(SPEC["baseline"]).fit(X[train], y[train]).predict(X[test])))
    b.append(score(y[test], build(SPEC["treatment"]).fit(X[train], y[train]).predict(X[test])))
print(json.dumps({
    "data_origin": "benchmark",
    "design": SPEC,
    "comparisons": [{
        "name": SPEC["dataset"] + " (" + SPEC["metric"] + ")",
        "labels": LABELS, "a": a, "b": b, "paired": True,
        "resampling": {"n_train": len(y) - n_test, "n_test": n_test},
    }],
}))
'''


def render_script(spec: ExperimentSpec) -> str:
    """Build the Python script. Only values that passed validation are inserted, as Python literals."""
    spec_literal = repr(spec.to_dict())
    labels_literal = repr([spec.baseline.describe(), spec.treatment.describe()])
    for literal in (spec_literal, labels_literal):
        ast.literal_eval(literal)                 # must be pure data: raises if anything else slipped in
    return _TEMPLATE.replace("__SPEC__", spec_literal).replace("__LABELS__", labels_literal)
