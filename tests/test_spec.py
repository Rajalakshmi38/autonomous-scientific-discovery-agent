"""Tests for experiment specifications (the model designs, vetted code writes the script).

The experiment template needs scikit-learn on THIS machine to be executed by the tests; it is installed
by requirements-dev.txt. The sandbox image has its own copy.
"""
import json
import subprocess
import sys

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.base import AgentError
from src.agents.code_executor import CodeExecutorAgent
from src.agents.hypothesis_generator import Hypothesis
from src.critic.critic import StatisticalCritic, Verdict
from src.experiments.spec import (
    DATASETS, MODEL_OPTIONS, SpecError, confounding_factors, default_spec, normalise_raw, options_text, parse_spec_text, render_script,
    design_mismatch_note, spec_from_text, validate_spec, validate_spec_tolerant,
)
from src.llm.stub_model import RuleBasedStubChatModel

HYP = Hypothesis("Scaling helps k-NN.", "r", "Compare k-NN with and without scaling on wine.", "raw")


def good(**over):
    spec = {"dataset": "iris",
            "baseline": {"model": "knn", "params": {"n_neighbors": 5}, "scale": False},
            "treatment": {"model": "knn", "params": {"n_neighbors": 5}, "scale": True},
            "metric": "accuracy", "n_resamples": 20}
    spec.update(over)
    return spec


# ------------------------------------------------------------------ validation
def test_a_good_spec_is_accepted_and_summarised():
    spec = validate_spec(good())
    assert spec.treatment.scale and not spec.baseline.scale
    assert spec.summary() == "iris: knn(n_neighbors=5) + scaling vs knn(n_neighbors=5), accuracy, 20 resamples"


@pytest.mark.parametrize("bad,fragment", [
    (good(dataset="imagenet"), "dataset"),
    (good(dataset=None), "dataset"),
    (good(metric="auc"), "metric"),
    (good(n_resamples=5), "n_resamples"),
    (good(n_resamples=500), "n_resamples"),
    (good(n_resamples=True), "n_resamples"),
    (good(n_resamples=12.5), "n_resamples"),
    (good(baseline="knn"), "baseline"),
    (good(baseline={"model": "xgboost"}), "not allowed"),
    (good(baseline={"model": "knn; import os", "params": {}}), "not allowed"),
    (good(baseline={"model": "knn", "params": {"n_neighbors": 7}}), "not allowed"),             # 7 not in the list
    (good(baseline={"model": "knn", "params": {"leaf_size": 3}}), "does not allow"),
    (good(baseline={"model": "knn", "params": {"n_neighbors": True}}), "not allowed"),
    (good(baseline={"model": "knn", "params": {"n_neighbors": [5]}}), "not allowed"),
    (good(baseline={"model": "knn", "params": {"n_neighbors": {"x": 1}}}), "not allowed"),
    (good(baseline={"model": "knn", "params": {"n_neighbors": "5); import os; (1"}}), "not allowed"),
    (good(baseline={"model": "knn", "scale": "yes"}), "scale"),
    (good(treatment={"model": "knn", "params": {"n_neighbors": 5}, "scale": False}), "identical"),
])
def test_bad_specs_are_rejected_with_a_helpful_message(bad, fragment):
    with pytest.raises(SpecError, match=fragment):
        validate_spec(bad)


def test_non_objects_are_rejected():
    for bad in (None, [], "iris", 5):
        with pytest.raises(SpecError):
            validate_spec(bad)


def test_none_is_allowed_where_the_option_list_contains_null():
    spec = validate_spec(good(baseline={"model": "decision_tree", "params": {"max_depth": None}}))
    assert spec.baseline.params == {"max_depth": None}


def test_values_are_replaced_by_our_own_canonical_values():
    spec = validate_spec(good(baseline={"model": "logistic_regression", "params": {"C": 1.0}}))
    assert spec.baseline.params["C"] == 1 and isinstance(spec.baseline.params["C"], int)


def test_unknown_extra_keys_are_ignored_not_executed():
    spec = validate_spec(good(note="TEST MODE", __import__="os"))
    assert "note" not in spec.to_dict() and "__import__" not in spec.to_dict()


# ------------------------------------------------------------------ parsing
def test_json_is_found_inside_chatty_replies():
    reply = "Sure! Here is the design:\n```json\n" + json.dumps(good()) + "\n```\nHope that helps."
    assert parse_spec_text(reply)["dataset"] == "iris"


def test_broken_braces_before_the_real_object_are_skipped():
    assert parse_spec_text("{oops} then " + json.dumps(good()))["metric"] == "accuracy"


def test_reply_without_json_is_an_error():
    with pytest.raises(SpecError, match="No JSON"):
        parse_spec_text("I would compare two models.")


# ------------------------------------------------------------------ prompt options and defaults
def test_options_text_lists_everything_the_validator_allows():
    text = options_text()
    assert all(d in text for d in DATASETS) and all(m in text for m in MODEL_OPTIONS)


def test_default_spec_is_valid():
    assert validate_spec(default_spec().to_dict()).summary() == default_spec().summary()


# ------------------------------------------------------------------ rendered script
def test_rendered_script_is_valid_python_and_contains_only_validated_values():
    code = render_script(validate_spec(good()))
    compile(code, "rendered", "exec")
    assert "knn" in code and "import os" not in code and "subprocess" not in code


@pytest.mark.parametrize("spec", [
    good(),
    good(dataset="wine", baseline={"model": "decision_tree", "params": {"max_depth": 2}},
         treatment={"model": "random_forest", "params": {"n_estimators": 10}}, metric="f1_macro", n_resamples=10),
])
def test_rendered_experiment_runs_and_the_critic_can_read_it(spec):
    pytest.importorskip("sklearn")
    code = render_script(validate_spec(spec))
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-500:]
    data = json.loads(run.stdout)
    comp = data["comparisons"][0]
    assert data["data_origin"] == "benchmark" and comp["paired"] and len(comp["a"]) == len(comp["b"]) == spec["n_resamples"]
    assert all(0.0 <= v <= 1.0 for v in comp["a"] + comp["b"])
    report = StatisticalCritic().evaluate(run.stdout)
    assert report.verdict in (Verdict.SIGNIFICANT, Verdict.INCONCLUSIVE)
    assert "RESAMPLED_DATA" in {f.code for f in report.findings} and "DATA_BENCHMARK" in {f.code for f in report.findings}


# ------------------------------------------------------------------ the agent in spec mode
def spec_reply(**over):
    return json.dumps(good(**over))


def test_agent_builds_a_script_from_a_valid_design():
    plan = CodeExecutorAgent(FakeListChatModel(responses=[spec_reply()]), mode="spec").prepare_experiment(HYP)
    assert plan.source == "model_spec" and plan.syntax_ok and plan.spec["dataset"] == "iris" and not plan.note
    assert "iris" in plan.design_summary


def test_agent_asks_again_after_an_invalid_design():
    fake = FakeListChatModel(responses=["not json at all", spec_reply(dataset="imagenet"), spec_reply(dataset="wine")])
    plan = CodeExecutorAgent(fake, mode="spec", spec_retries=2).prepare_experiment(HYP)
    assert plan.source == "model_spec" and plan.spec["dataset"] == "wine"


def test_agent_uses_fixed_keyword_rules_when_the_models_design_is_hopeless():
    plan = CodeExecutorAgent(FakeListChatModel(responses=["no json"]), mode="spec", spec_retries=1).prepare_experiment(HYP)
    assert plan.source == "keyword_spec" and plan.syntax_ok
    assert plan.spec["treatment"]["scale"] is True and plan.spec["dataset"] == "wine"        # HYP: scaling helps k-NN on wine
    assert "no language model" in plan.note and "Check that it fits" in plan.note


def test_agent_falls_back_to_a_labelled_default_when_nothing_in_the_hypothesis_helps():
    vague = Hypothesis("Graph attention improves node embeddings.", "r", "Train and compare.", "raw")
    plan = CodeExecutorAgent(FakeListChatModel(responses=["no json"]), mode="spec", spec_retries=1).prepare_experiment(vague)
    assert plan.source == "default_spec" and plan.syntax_ok
    assert "does NOT test the hypothesis" in plan.note


def test_small_slips_in_the_models_design_are_repaired_and_reported():
    sloppy = json.dumps({"dataset": "Wine", "metric": "F1", "n_resamples": 100,
                         "baseline": {"model": "Random Forest", "params": {"trees": 7, "random_state": 1}, "scale": "yes"},
                         "treatment": {"model": "k-Nearest Neighbors", "params": {"k": 7}, "scale": "true"}})
    plan = CodeExecutorAgent(FakeListChatModel(responses=[sloppy]), mode="spec").prepare_experiment(HYP)
    assert plan.source == "model_spec"
    assert plan.design_summary == "wine: knn(n_neighbors=5) + scaling vs random_forest(n_estimators=5) + scaling, f1_macro, 30 resamples"
    assert "adjusted to allowed options" in plan.note and "n_neighbors 7 -> 5" in plan.note


def test_spec_mode_does_not_support_rewriting_code():
    assert not CodeExecutorAgent(RuleBasedStubChatModel(), mode="spec").supports_repair
    assert CodeExecutorAgent(RuleBasedStubChatModel(), mode="free").supports_repair


def test_model_failure_becomes_an_agent_error():
    class Broken(RuleBasedStubChatModel):
        def _generate(self, *a, **k):
            raise ConnectionError("ollama not running")
    with pytest.raises(AgentError, match="design model call failed"):
        CodeExecutorAgent(Broken(), mode="spec").prepare_experiment(HYP)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        CodeExecutorAgent(RuleBasedStubChatModel(), mode="magic")


def test_the_prompt_templates_render_with_their_braces_intact():
    from langchain_core.prompts import ChatPromptTemplate
    from src.agents.code_executor import SPEC_SYSTEM
    message = ChatPromptTemplate.from_messages([("system", SPEC_SYSTEM)]).format_messages(options="OPTIONS")[0].content
    assert '"dataset": "iris"' in message and "OPTIONS" in message


# ------------------------------------------------------------------ tolerant reading
def test_normalise_changes_nothing_in_a_form_that_is_already_valid():
    cleaned, notes = normalise_raw(good())
    assert notes == [] and validate_spec(cleaned).summary() == validate_spec(good()).summary()


@pytest.mark.parametrize("name,expected", [
    ("Random Forest", "random_forest"), ("RandomForestClassifier", "random_forest"), ("random-forest", "random_forest"),
    ("k-NN", "knn"), ("KNeighborsClassifier", "knn"), ("Logistic Regression", "logistic_regression"),
    ("Decision Tree", "decision_tree"), ("Naive Bayes", "naive_bayes"), ("SVC", "svm"), ("Support Vector Machine", "svm"),
])
def test_model_names_are_recognised(name, expected):
    spec, _ = validate_spec_tolerant(good(baseline={"model": name}, treatment={"model": "svm" if expected != "svm" else "knn"}))
    assert spec.baseline.model == expected


def test_numbers_snap_to_the_nearest_allowed_option():
    spec, notes = validate_spec_tolerant(good(baseline={"model": "knn", "params": {"n_neighbors": 7}},
                                              treatment={"model": "knn", "params": {"n_neighbors": 20}}))
    assert spec.baseline.params["n_neighbors"] == 5 and spec.treatment.params["n_neighbors"] == 25
    assert any("7 -> 5" in n for n in notes) and any("20 -> 25" in n for n in notes)


def test_a_tie_between_two_allowed_values_goes_to_the_lower_one():
    from src.experiments.spec import _snap
    assert _snap(8, [1, 3, 5, 11, 25]) == 5            # 8 is 3 away from both 5 and 11


def test_slips_that_cannot_be_repaired_are_still_rejected():
    for bad in (good(dataset="imagenet"), good(baseline={"model": "xgboost"}), good(baseline=None)):
        with pytest.raises(SpecError):
            validate_spec_tolerant(bad)
    same = good(baseline={"model": "knn", "params": {"n_neighbors": 7}}, treatment={"model": "knn", "params": {"n_neighbors": 5}})
    with pytest.raises(SpecError, match="identical"):          # 7 snaps to 5, so there is nothing left to compare
        validate_spec_tolerant(same)


def test_repaired_forms_still_only_contain_allowed_values():
    spec, _ = validate_spec_tolerant(good(baseline={"model": "Random Forest", "params": {"max_depth": 3, "n_estimators": "50"}},
                                          treatment={"model": "svm", "params": {"kernel": "RBF"}}))
    code = render_script(spec)
    compile(code, "r", "exec")
    assert "RBF" not in code and "'rbf'" in code


# ------------------------------------------------------------------ rule-based fallback
@pytest.mark.parametrize("text,summary", [
    ("Scaling the features improves k-NN accuracy on wine.", "wine: knn + scaling vs knn, accuracy, 20 resamples"),
    ("Standardisation helps an SVM.", "wine: svm + scaling vs svm, accuracy, 20 resamples"),
    ("Random forests outperform logistic regression on the iris data.", "iris: random_forest vs logistic_regression, accuracy, 20 resamples"),
    ("More trees in a random forest help.", "iris: random_forest(n_estimators=100) vs random_forest(n_estimators=10), accuracy, 20 resamples"),
    ("An SVM beats a decision tree on digits.", "digits: svm vs decision_tree, accuracy, 20 resamples"),
])
def test_keyword_rules_build_a_sensible_design(text, summary):
    spec, why = spec_from_text(text)
    assert spec is not None and spec.summary() == summary and why


@pytest.mark.parametrize("text", [
    "Graph attention improves node embeddings.", "The basic device conceals ongoing news.", "", "Wine is nice but there is no method.",
])
def test_keyword_rules_say_so_when_nothing_matches(text):
    spec, why = spec_from_text(text)
    assert spec is None and "no machine-learning" in why


def test_keyword_designs_are_always_valid_and_renderable():
    for text in ("Scaling helps naive bayes", "decision tree depth", "svm vs knn", "logistic regression regularisation"):
        spec, _ = spec_from_text(text)
        compile(render_script(validate_spec(spec.to_dict())), "r", "exec")


# ------------------------------------------------------------------ a fair experiment changes exactly ONE thing
def two(base, treat):
    return {"dataset": "iris", "baseline": base, "treatment": treat}


@pytest.mark.parametrize("base,treat,factors", [
    ({"model": "knn"}, {"model": "knn", "scale": True}, ["feature scaling"]),
    ({"model": "knn", "params": {"n_neighbors": 3}}, {"model": "knn", "params": {"n_neighbors": 11}}, ["the parameter n_neighbors"]),
    ({"model": "svm"}, {"model": "random_forest"}, ["the model"]),
    ({"model": "random_forest", "params": {"n_estimators": 50}}, {"model": "logistic_regression", "params": {"C": 1}, "scale": True},
     ["the model", "feature scaling"]),                                       # the design from a real 3B-model run
    ({"model": "random_forest", "params": {"n_estimators": 10, "max_depth": 2}}, {"model": "random_forest", "params": {"n_estimators": 100, "max_depth": 5}},
     ["the parameter max_depth", "the parameter n_estimators"]),
])
def test_the_factors_that_differ_are_counted(base, treat, factors):
    assert confounding_factors(validate_spec(two(base, treat))) == factors


def test_designs_that_change_several_things_are_rejected_with_advice():
    with pytest.raises(SpecError, match="the model and feature scaling.*exactly ONE thing"):
        validate_spec_tolerant(two({"model": "random_forest"}, {"model": "logistic_regression", "scale": True}))
    with pytest.raises(SpecError, match="exactly ONE thing"):
        validate_spec_tolerant(two({"model": "random_forest", "params": {"n_estimators": 10, "max_depth": 2}},
                                   {"model": "random_forest", "params": {"n_estimators": 100, "max_depth": 5}}))
    validate_spec_tolerant(two({"model": "random_forest"}, {"model": "logistic_regression", "scale": True}), single_factor=False)


def test_the_model_is_asked_again_when_its_design_changes_two_things():
    confounded = json.dumps(two({"model": "random_forest"}, {"model": "logistic_regression", "scale": True}))
    fixed = json.dumps(two({"model": "knn"}, {"model": "knn", "scale": True}))
    plan = CodeExecutorAgent(FakeListChatModel(responses=[confounded, fixed]), mode="spec", spec_retries=1).prepare_experiment(HYP)
    assert plan.source == "model_spec" and plan.design_summary == "iris: knn + scaling vs knn, accuracy, 20 resamples"


def test_keyword_rules_take_over_when_the_model_keeps_confounding():
    confounded = json.dumps(two({"model": "random_forest"}, {"model": "logistic_regression", "scale": True}))
    plan = CodeExecutorAgent(FakeListChatModel(responses=[confounded]), mode="spec", spec_retries=1).prepare_experiment(HYP)
    assert plan.source == "keyword_spec" and len(confounding_factors(validate_spec(plan.spec))) == 1


def test_every_fallback_design_changes_exactly_one_thing():
    from src.experiments.spec import _ONE_MODEL_SETTINGS
    for text in ("Scaling helps naive bayes", "svm vs knn", "random forest", "decision tree depth", "logistic regression", "k-NN"):
        spec, _ = spec_from_text(text)
        assert len(confounding_factors(spec)) == 1, text
    assert len(confounding_factors(default_spec())) == 1
    assert set(_ONE_MODEL_SETTINGS) >= {"random_forest", "knn", "svm"}


# ------------------------------------------------------------------ the design must fit what the hypothesis says
@pytest.mark.parametrize("text, summary", [
    ("Increasing the number of trees in a Random Forest from 10 to 100 raises the score [arxiv:2606.03549].",
     "iris: random_forest(n_estimators=100) vs random_forest(n_estimators=10), accuracy, 20 resamples"),
    ("A k-NN with k=11 beats a k-NN with k=1 on wine.",
     "wine: knn(n_neighbors=11) vs knn(n_neighbors=1), accuracy, 20 resamples"),
])
def test_keyword_rules_read_two_values_of_a_setting(text, summary):
    spec, why = spec_from_text(text)
    assert spec is not None and spec.summary() == summary and "two values" in why
    assert len(confounding_factors(spec)) == 1


def test_numbers_inside_a_citation_are_not_used_as_values():
    spec, _ = spec_from_text("More trees in a random forest help [arxiv:2606.03549].")
    assert spec.summary() == "iris: random_forest(n_estimators=100) vs random_forest(n_estimators=10), accuracy, 20 resamples"


def test_the_hypothesis_sentence_beats_a_scaling_word_in_the_test_plan():
    hyp = Hypothesis("Increasing the number of trees in a Random Forest from 10 to 100 raises accuracy.", "r",
                     "Use 20 resamples; scaling is not needed.", "raw")
    plan = CodeExecutorAgent(FakeListChatModel(responses=["no json"]), mode="spec", spec_retries=1).prepare_experiment(hyp)
    assert plan.source == "keyword_spec"
    assert plan.spec["baseline"]["params"] == {"n_estimators": 10} and plan.spec["treatment"]["params"] == {"n_estimators": 100}
    assert not plan.spec["baseline"]["scale"] and not plan.spec["treatment"]["scale"]


def test_a_test_plan_is_used_only_when_the_hypothesis_sentence_has_nothing():
    hyp = Hypothesis("The new method is better.", "r", "Compare k-NN with and without scaling on wine.", "raw")
    plan = CodeExecutorAgent(FakeListChatModel(responses=["no json"]), mode="spec", spec_retries=1).prepare_experiment(hyp)
    assert plan.source == "keyword_spec" and plan.spec["treatment"]["scale"] is True
    assert "test plan" in plan.note


def _design(base, treat):
    return validate_spec({"dataset": "iris", "baseline": base, "treatment": treat})


def test_mismatch_note_when_the_hypothesis_names_no_testable_method():
    design = _design({"model": "logistic_regression"}, {"model": "logistic_regression", "scale": True})
    note = design_mismatch_note(design, "Weighted STreeD will outperform standard STreeD on iris.")
    assert "names none of the methods" in note and "does not mention scaling" in note


def test_mismatch_note_when_the_design_uses_a_method_the_hypothesis_does_not_name():
    design = _design({"model": "random_forest"}, {"model": "logistic_regression"})
    note = design_mismatch_note(design, "A random forest with 100 trees scores higher than one with 10 trees.")
    assert "uses logistic_regression" in note


def test_no_mismatch_note_when_the_design_fits():
    design = _design({"model": "random_forest", "params": {"n_estimators": 10}},
                     {"model": "random_forest", "params": {"n_estimators": 100}})
    assert design_mismatch_note(design, "A random forest with 100 trees scores higher than one with 10 trees.") == ""


def test_the_agent_warns_when_the_models_valid_design_does_not_fit_the_hypothesis():
    hyp = Hypothesis("A random forest with 100 trees scores higher than one with 10 trees.", "r", "plan", "raw")
    reply = json.dumps(good(baseline={"model": "logistic_regression", "params": {}, "scale": False},
                            treatment={"model": "logistic_regression", "params": {}, "scale": True}))
    plan = CodeExecutorAgent(FakeListChatModel(responses=[reply]), mode="spec").prepare_experiment(hyp)
    assert plan.source == "model_spec" and "Possible mismatch" in plan.note and "random_forest" in plan.note


# ------------------------------------------------------------------ the hypothesis prompt must match what the sandbox allows
def test_the_hypothesis_scope_lists_the_real_allowed_values_and_does_not_push_one_method():
    from src.agents.hypothesis_generator import EXPERIMENT_SCOPE
    from src.experiments.spec import MODEL_OPTIONS
    trees = MODEL_OPTIONS["random_forest"]["n_estimators"]
    ks = MODEL_OPTIONS["knn"]["n_neighbors"]
    assert "number of trees " + ", ".join(str(x) for x in trees[:-1]) + " or " + str(trees[-1]) in EXPERIMENT_SCOPE
    assert "k of " + ", ".join(str(x) for x in ks[:-1]) + " or " + str(ks[-1]) in EXPERIMENT_SCOPE
    for dataset in ("iris", "wine", "breast_cancer", "digits"):
        assert dataset in EXPERIMENT_SCOPE
    assert "do not default to random forests" in EXPERIMENT_SCOPE
    assert "100 trees scores higher than one with 10 trees" not in EXPERIMENT_SCOPE       # a single example gets copied
    assert EXPERIMENT_SCOPE.count("scores higher") >= 3 and "k-nearest neighbors with k=11" in EXPERIMENT_SCOPE
    assert "do not claim" in EXPERIMENT_SCOPE and "predicts anything else" in EXPERIMENT_SCOPE


# ------------------------------------------------------------------ found in the second holdout evaluation
SVM_FORM = {"dataset": "breast_cancer", "metric": "f1_macro", "n_resamples": 20,
            "baseline": {"model": "svm", "params": {"kernel": "rbf", "scale": False}},
            "treatment": {"model": "svm", "params": {"kernel": "linear", "scale": True}}}


def test_a_scaling_flag_written_inside_params_is_moved_to_the_scale_setting():
    raw = {"dataset": "iris", "baseline": {"model": "svm", "params": {"scale": False}},
           "treatment": {"model": "svm", "params": {"scale": True}}}
    spec, notes = validate_spec_tolerant(raw)
    assert spec.baseline.scale is False and spec.treatment.scale is True
    assert spec.treatment.params == {} and any("moved to treatment.scale" in n for n in notes)
    assert not any("ignored" in n for n in notes)


def test_an_explicit_scale_setting_wins_over_a_flag_inside_params():
    raw = {"dataset": "iris", "baseline": {"model": "svm", "scale": False, "params": {"scale": True}},
           "treatment": {"model": "svm", "scale": True}}
    spec, notes = validate_spec_tolerant(raw)
    assert spec.baseline.scale is False and any("baseline.params.scale ignored" in n for n in notes)


def test_a_design_that_changes_the_kernel_and_scaling_together_falls_back_to_the_hypothesis():
    """The real case: the model changed the kernel AND scaling. The one-thing rule refuses it, and the fallback reads
    the hypothesis sentence, which is about scaling."""
    hyp = Hypothesis("A feature-scaled SVM scores higher macro-F1 than an SVM without feature scaling on the breast_cancer dataset.",
                     "r", "plan", "raw")
    reply = json.dumps(SVM_FORM)
    plan = CodeExecutorAgent(FakeListChatModel(responses=[reply] * 4), mode="spec", spec_retries=2).prepare_experiment(hyp)
    assert plan.source == "keyword_spec"
    spec = plan.spec
    assert spec["baseline"]["model"] == spec["treatment"]["model"] == "svm"
    assert spec["baseline"]["scale"] is False and spec["treatment"]["scale"] is True
    assert spec["baseline"]["params"] == spec["treatment"]["params"]


def test_mismatch_note_when_the_hypothesis_is_about_scaling_but_the_arms_do_not_differ_in_scaling():
    design = _design({"model": "svm", "params": {"kernel": "rbf"}}, {"model": "svm", "params": {"kernel": "linear"}})
    note = design_mismatch_note(design, "A feature-scaled SVM scores higher macro-F1 than an SVM without feature scaling.")
    assert "about feature scaling" in note and "do not differ in scaling" in note
    fitting = _design({"model": "svm"}, {"model": "svm", "scale": True})
    assert design_mismatch_note(fitting, "A feature-scaled SVM scores higher macro-F1 than an SVM without feature scaling.") == ""
