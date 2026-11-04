"""Sandbox, repair-loop and execution-workflow tests.

Three layers, so you always know what a green test proves:
  1. Unit tests: command construction, error classification, loop logic (no processes).
  2. FAKE-DOCKER tests: run our real subprocess code against tests/fake_docker.py. They prove our
     handling of stdin, timeouts, output caps, cleanup and failures. The fake provides NO isolation.
  3. REAL-DOCKER tests (marker `docker`): skipped unless Docker is running and the image is built
     (python -m src.main sandbox-build). Only these show that the restrictions actually work.

SAFETY: never run the probe scripts in self_test.PROBES through the fake docker: the fake executes
code inside the test process, and some probes deliberately eat memory or fork.
"""
import json
import sys
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from config.settings import ConfigError, load_settings
from src.agents.base import AgentError
from src.agents.code_executor import CodeExecutorAgent, ExperimentPlan
from src.agents.hypothesis_generator import Hypothesis, HypothesisGeneratorAgent
from src.agents.literature_reviewer import LiteratureReviewerAgent
from src.execution.feedback_loop import run_with_repair, summarize_error
from src.execution.sandbox import (
    DockerSandbox, ExecutionResult, SandboxConfig, SandboxStatus, Status, build_run_command, classify_error,
    missing_module,
)
from src.execution.self_test import PROBES, run_self_test
from src.ingestion.paper_store import PaperStore
from src.llm.stub_model import RuleBasedStubChatModel
from src.models import Paper
from src.retrieval.embeddings import HashingEmbedder
from src.retrieval.retriever import PaperRetriever
from src.retrieval.vector_store import NumpyVectorStore
from src.workflow import ResearchWorkflow

FAKE_DOCKER = str(Path(__file__).resolve().parent / "fake_docker.py")
HYP = Hypothesis("statement", "rationale", "plan", "raw")


def fake_sandbox(**overrides):
    cfg = dict(docker_command=(sys.executable, FAKE_DOCKER), timeout_seconds=5, max_output_bytes=2000)
    cfg.update(overrides)
    return DockerSandbox(SandboxConfig(**cfg))


@pytest.fixture
def docker_log(monkeypatch, tmp_path):
    log = tmp_path / "docker_calls.log"
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
    return log


# =========================================================== 1. unit tests
def test_run_command_contains_every_restriction_and_no_host_mounts():
    cmd = build_run_command(SandboxConfig(memory_mb=128, pids_limit=32), "asda-sbx-test")
    joined = " ".join(cmd)
    for required in ("--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
                     "--memory 128m", "--memory-swap 128m", "--pids-limit 32", "--user 10001:10001",
                     "--pull never"):
        assert required in joined
    for forbidden in ("-v ", "--volume", "--mount", "--privileged", "--network host", "--cap-add"):
        assert forbidden not in joined
    assert cmd[-3:] == ["asda-sandbox:latest", "python", "-"]      # script arrives on stdin


def test_error_classification():
    assert classify_error("ModuleNotFoundError: No module named 'torch'") == "missing_dependency"
    assert missing_module("ModuleNotFoundError: No module named 'torch_geometric.nn'") == "torch_geometric.nn"
    assert classify_error("  File x\nSyntaxError: invalid syntax") == "syntax_error"
    assert classify_error("Traceback (most recent call last):\nValueError: x") == "runtime_error"
    assert classify_error("") == "unknown_error"


def test_result_flags():
    assert ExecutionResult(Status.SUCCESS).ok
    assert ExecutionResult(Status.FAILED).repairable and ExecutionResult(Status.TIMEOUT).repairable
    assert not ExecutionResult(Status.SANDBOX_UNAVAILABLE).repairable
    assert not ExecutionResult(Status.SANDBOX_ERROR).repairable


def test_settings_for_sandbox_are_validated():
    with pytest.raises(ConfigError):
        load_settings(env={"MAX_REPAIR_RETRIES": "9"}, load_env_file=False)
    with pytest.raises(ConfigError):
        load_settings(env={"SANDBOX_MEMORY_MB": "10"}, load_env_file=False)
    cfg = SandboxConfig.from_settings(load_settings(env={"DOCKER_COMMAND": "wsl docker"}, load_env_file=False))
    assert cfg.docker_command == ("wsl", "docker") and cfg.memory_mb == 512


# =========================================================== 2. fake-docker tests (real subprocess handling)
def test_successful_execution_captures_stdout():
    r = fake_sandbox().run_code("import json\nprint(json.dumps({'answer': 42}))")
    assert r.status == Status.SUCCESS and r.exit_code == 0
    assert json.loads(r.stdout) == {"answer": 42}


def test_failed_execution_captures_error():
    r = fake_sandbox().run_code("print('before')\nraise ValueError('boom')")
    assert r.status == Status.FAILED and r.exit_code == 1
    assert r.error_kind == "runtime_error" and "ValueError: boom" in r.stderr and "before" in r.stdout


def test_missing_dependency_is_reported_clearly():
    r = fake_sandbox().run_code("import module_that_does_not_exist_xyz")
    assert r.status == Status.FAILED and r.error_kind == "missing_dependency"
    assert "module_that_does_not_exist_xyz" in r.message


def test_syntax_error_is_reported():
    r = fake_sandbox().run_code("def broken(:\n    pass")
    assert r.status == Status.FAILED and r.error_kind == "syntax_error"


def test_timeout_kills_the_container_and_cleans_up(docker_log):
    r = fake_sandbox().run_code("while True:\n    pass", timeout_seconds=2)
    assert r.status == Status.TIMEOUT and r.error_kind == "timeout"
    assert 1.5 < r.duration_seconds < 15
    calls = docker_log.read_text()
    assert f"kill {r.container_name}" in calls        # the CONTAINER is stopped, not just the client
    assert f"rm -f {r.container_name}" in calls       # and removed
    assert fake_sandbox().leftover_containers() == []


def test_out_of_memory_is_detected():
    r = fake_sandbox().run_code("#FAKE_OOM\nprint('never printed')")
    assert r.status == Status.OUT_OF_MEMORY and r.error_kind == "out_of_memory"


def test_runaway_output_is_capped():
    r = fake_sandbox(max_output_bytes=2000).run_code("print('x' * 1_000_000)")
    assert r.status == Status.SUCCESS and r.output_truncated and len(r.stdout) == 2000


def test_container_is_removed_after_normal_runs(docker_log):
    r = fake_sandbox().run_code("print(1)")
    assert f"rm -f {r.container_name}" in docker_log.read_text()


@pytest.mark.parametrize("code", ["", "   \n", None])
def test_empty_code_is_rejected_without_starting_docker(code, docker_log):
    r = fake_sandbox().run_code(code)
    assert r.status == Status.FAILED and r.error_kind == "invalid_input"
    assert not docker_log.exists()


def test_oversized_code_is_rejected(docker_log):
    assert fake_sandbox(max_code_bytes=100).run_code("x = 1\n" * 100).error_kind == "invalid_input"


# ---- unavailable external service: Docker itself
def test_docker_not_installed_is_reported_not_crashed():
    sb = DockerSandbox(SandboxConfig(docker_command=("docker-command-that-does-not-exist",)))
    assert sb.run_code("print(1)").status == Status.SANDBOX_UNAVAILABLE
    status = sb.check_available()
    assert not status.ok and "not found" in status.message


def test_docker_engine_not_running(monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_DOWN", "1")
    sb = fake_sandbox()
    status = sb.check_available()
    assert status.ok is False and status.docker_found and not status.engine_running
    assert "engine is not running" in status.message
    r = sb.run_code("#FAKE_DOCKER_DOWN")
    assert r.status == Status.SANDBOX_UNAVAILABLE and r.error_kind == "docker_not_running"


def test_image_not_built(monkeypatch):
    monkeypatch.setenv("FAKE_NO_IMAGE", "1")
    status = fake_sandbox().check_available()
    assert status.ok is False and status.engine_running and not status.image_present
    assert "sandbox-build" in status.message
    r = fake_sandbox().run_code("#FAKE_NO_IMAGE")
    assert r.status == Status.SANDBOX_UNAVAILABLE and r.error_kind == "image_missing"


def test_available_when_docker_and_image_present():
    status = fake_sandbox().check_available()
    assert status.ok is True and status.docker_found and status.engine_running and status.image_present


# =========================================================== 3. repair loop
class ScriptedSandbox:
    def __init__(self, results):
        self.results, self.codes = list(results), []

    def run_code(self, code, timeout_seconds=None):
        self.codes.append(code)
        return self.results.pop(0)


class ScriptedRepairer:
    def __init__(self, fixes):
        self.fixes, self.errors = list(fixes), []

    def repair_experiment(self, hypothesis, code, error_text):
        self.errors.append(error_text)
        fix = self.fixes.pop(0)
        if isinstance(fix, Exception):
            raise fix
        return ExperimentPlan(code=fix, syntax_ok=True)


FAIL = ExecutionResult(Status.FAILED, 1, stderr="Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'",
                       error_kind="missing_dependency", message="needs a library")
OK = ExecutionResult(Status.SUCCESS, 0, stdout="{}")


def test_loop_stops_immediately_on_success():
    sb, rep = ScriptedSandbox([OK]), ScriptedRepairer([])
    out = run_with_repair(sb, rep, HYP, "code1", max_repairs=2)
    assert out.final_status == Status.SUCCESS and len(out.attempts) == 1 and rep.errors == []


def test_loop_repairs_then_succeeds():
    sb, rep = ScriptedSandbox([FAIL, OK]), ScriptedRepairer(["code2"])
    out = run_with_repair(sb, rep, HYP, "code1", max_repairs=2)
    assert out.final_status == Status.SUCCESS and sb.codes == ["code1", "code2"]
    assert "'torch' is NOT installed" in rep.errors[0]            # the model is told what went wrong


def test_loop_is_bounded_by_the_repair_limit():
    sb = ScriptedSandbox([FAIL, FAIL, FAIL, FAIL])
    rep = ScriptedRepairer(["c2", "c3", "c4"])
    out = run_with_repair(sb, rep, HYP, "c1", max_repairs=2)
    assert out.final_status == Status.FAILED
    assert len(out.attempts) == 3 and len(rep.errors) == 2         # 1 run + 2 repairs, never more
    assert "repair limit" in out.attempts[-1].note


def test_zero_repairs_means_a_single_run():
    sb, rep = ScriptedSandbox([FAIL]), ScriptedRepairer([])
    out = run_with_repair(sb, rep, HYP, "c1", max_repairs=0)
    assert len(out.attempts) == 1 and rep.errors == []


@pytest.mark.parametrize("status", [Status.SANDBOX_UNAVAILABLE, Status.SANDBOX_ERROR])
def test_loop_does_not_retry_sandbox_problems(status):
    sb, rep = ScriptedSandbox([ExecutionResult(status, message="docker down")]), ScriptedRepairer([])
    out = run_with_repair(sb, rep, HYP, "c1", max_repairs=3)
    assert len(out.attempts) == 1 and out.final_status == status and rep.errors == []


def test_loop_retries_timeouts_and_memory_failures():
    sb = ScriptedSandbox([ExecutionResult(Status.TIMEOUT, message="slow"), ExecutionResult(Status.OUT_OF_MEMORY), OK])
    out = run_with_repair(sb, ScriptedRepairer(["c2", "c3"]), HYP, "c1", max_repairs=2)
    assert out.final_status == Status.SUCCESS and len(out.attempts) == 3


def test_loop_stops_if_model_returns_identical_code():
    out = run_with_repair(ScriptedSandbox([FAIL]), ScriptedRepairer(["c1"]), HYP, "c1", max_repairs=3)
    assert len(out.attempts) == 1 and "same code" in out.attempts[0].note


def test_loop_stops_if_the_repair_model_fails():
    out = run_with_repair(ScriptedSandbox([FAIL]), ScriptedRepairer([AgentError("ollama down")]), HYP, "c1", 2)
    assert len(out.attempts) == 1 and "could not repair" in out.attempts[0].note


def test_error_summary_is_short_and_keeps_the_end_of_stderr():
    long_err = "x" * 5000 + "\nFINAL_ERROR_LINE"
    text = summarize_error(ExecutionResult(Status.FAILED, 1, stderr=long_err, message="m"), max_chars=300)
    assert "FINAL_ERROR_LINE" in text and len(text) < 700


def test_repair_loop_end_to_end_through_real_subprocess_handling():
    """Broken code -> real failure (missing module) -> model 'fix' -> real success."""
    fake_llm = FakeListChatModel(responses=[
        "```python\nimport module_that_does_not_exist_xyz\n```",
        "```python\nimport json\nprint(json.dumps({'fixed': True}))\n```",
    ])
    agent = CodeExecutorAgent(fake_llm)
    plan = agent.prepare_experiment(HYP)
    out = run_with_repair(fake_sandbox(), agent, HYP, plan.code, max_repairs=2)
    assert out.final_status == Status.SUCCESS and len(out.attempts) == 2
    assert out.attempts[0].result.error_kind == "missing_dependency"
    assert json.loads(out.attempts[1].result.stdout) == {"fixed": True}


# =========================================================== 4. workflow integration
@pytest.fixture
def workflow_parts(tmp_path):
    corpus = [Paper("arxiv", "1", "Graph neural networks", "Message passing on graph neural networks.", url="u1")]
    store = PaperStore(tmp_path / "p.db")
    store.upsert_many(corpus)
    retriever = PaperRetriever(store, HashingEmbedder(), NumpyVectorStore())
    retriever.index_papers(corpus)
    llm = RuleBasedStubChatModel()
    yield retriever, llm
    store.close()


def make_workflow(parts, sandbox):
    retriever, llm = parts
    return ResearchWorkflow(retriever, LiteratureReviewerAgent(llm), HypothesisGeneratorAgent(llm),
                            CodeExecutorAgent(llm), llm_mode="TEST MODE", top_k=3, sandbox=sandbox, max_repairs=1)


def test_full_pipeline_with_execution_test_mode(workflow_parts, tmp_path):
    result = make_workflow(workflow_parts, fake_sandbox()).run("graph neural networks")
    assert result.status == "COMPLETED" and result.execution_status == Status.SUCCESS
    stdout = result.execution.final_result.stdout
    assert json.loads(stdout)["data_origin"] == "synthetic"         # the mock script labels its data synthetic
    assert result.critic.verdict == "INCONCLUSIVE"                  # same distribution on both sides: no effect by design
    assert result.critic.data_origin == "synthetic"
    saved = json.loads(result.save(tmp_path / "runs").read_text(encoding="utf-8"))
    assert saved["execution"]["attempt_count"] == 1 and saved["llm_mode"] == "TEST MODE"
    assert saved["sources_cited_by_hypothesis"][0]["id"] == "arxiv:1"           # traceable id ...
    assert saved["sources_cited_by_hypothesis"][0]["url"] == "u1"               # ... and url


def test_pipeline_reports_unavailable_sandbox_and_never_runs_on_host(workflow_parts, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_DOWN", "1")
    result = make_workflow(workflow_parts, fake_sandbox()).run("graph neural networks")
    assert result.status == "COMPLETED" and result.execution_status == Status.SANDBOX_UNAVAILABLE
    assert result.execution.attempts == []
    assert result.critic is None
    assert any("NOT executed" in w for w in result.warnings)


def test_pipeline_without_sandbox_does_not_execute(workflow_parts):
    result = make_workflow(workflow_parts, None).run("graph neural networks")
    assert result.execution is None and result.execution_status == Status.NOT_RUN
    assert result.critic is None
    assert any("not executed" in w for w in result.warnings)


def test_pipeline_surfaces_failed_execution_honestly(workflow_parts):
    retriever, _ = workflow_parts
    broken = FakeListChatModel(responses=["```python\nraise RuntimeError('nope')\n```"])
    wf = ResearchWorkflow(retriever, LiteratureReviewerAgent(RuleBasedStubChatModel()),
                          HypothesisGeneratorAgent(RuleBasedStubChatModel()), CodeExecutorAgent(broken),
                          llm_mode="TEST MODE", sandbox=fake_sandbox(), max_repairs=1)
    result = wf.run("graph neural networks")
    assert result.execution_status == Status.FAILED          # the identical 'fix' is detected and not retried
    assert any("ended with FAILED" in w for w in result.warnings)
    assert result.critic.verdict == "NO_RESULT"              # nothing ran successfully, so nothing to check


# =========================================================== 5. REAL Docker tests (skipped without Docker + image)
@pytest.fixture(scope="module")
def real_sandbox():
    sandbox = DockerSandbox(SandboxConfig())
    status = sandbox.check_available()
    if not status.ok:
        pytest.skip(f"real Docker sandbox not ready: {status.message}")
    return sandbox


@pytest.fixture(scope="module")
def probe_results(real_sandbox):
    return {r.name: r for r in run_self_test(real_sandbox)}


@pytest.mark.docker
@pytest.mark.parametrize("name", [p.name for p in PROBES] + ["containers_cleaned_up"])
def test_real_sandbox_restriction(probe_results, name):
    result = probe_results[name]
    assert result.passed, f"{name}: {result.what} -> {result.detail}"


@pytest.mark.docker
def test_real_docker_success_failure_and_missing_dependency(real_sandbox):
    ok = real_sandbox.run_code("import json\nprint(json.dumps({'v': 1}))")
    assert ok.status == Status.SUCCESS and json.loads(ok.stdout) == {"v": 1}
    bad = real_sandbox.run_code("raise ValueError('boom')")
    assert bad.status == Status.FAILED and "ValueError" in bad.stderr
    missing = real_sandbox.run_code("import torch")
    assert missing.status == Status.FAILED and missing.error_kind == "missing_dependency"
    assert real_sandbox.run_code("import numpy\nprint(numpy.__version__)").status == Status.SUCCESS


@pytest.mark.docker
def test_real_docker_has_the_scientific_libraries_and_offline_datasets(real_sandbox):
    code = ("import numpy, scipy, pandas, sklearn\n"
            "from sklearn.datasets import load_iris\n"
            "print(load_iris().data.shape)\n")
    r = real_sandbox.run_code(code)
    assert r.status == Status.SUCCESS and r.stdout.strip() == "(150, 4)", r.stderr


@pytest.mark.docker
def test_real_docker_timeout(real_sandbox):
    r = real_sandbox.run_code("while True:\n    pass", timeout_seconds=4)
    assert r.status == Status.TIMEOUT and real_sandbox.leftover_containers() == []


@pytest.mark.docker
def test_real_docker_repair_loop_end_to_end(real_sandbox):
    fake_llm = FakeListChatModel(responses=[
        "```python\nimport torch\nprint(torch.__version__)\n```",
        "```python\nimport json\nprint(json.dumps({'fixed': True}))\n```",
    ])
    agent = CodeExecutorAgent(fake_llm)
    plan = agent.prepare_experiment(HYP)
    out = run_with_repair(real_sandbox, agent, HYP, plan.code, max_repairs=2)
    assert out.final_status == Status.SUCCESS and len(out.attempts) == 2


# =========================================================== 6. spec mode (Week 3): vetted templates, memory, unload hook
from src.memory.store import MemoryStore


def test_loop_does_not_rewrite_vetted_template_experiments():
    agent = CodeExecutorAgent(RuleBasedStubChatModel(), mode="spec")
    out = run_with_repair(ScriptedSandbox([FAIL]), agent, HYP, "rendered code", max_repairs=3)
    assert len(out.attempts) == 1 and "vetted template" in out.attempts[0].note


def spec_workflow(parts, tmp_path, executor=None, hook=None):
    retriever, llm = parts
    return ResearchWorkflow(
        retriever, LiteratureReviewerAgent(llm), HypothesisGeneratorAgent(llm, scope="SCOPE"),
        executor or CodeExecutorAgent(llm, mode="spec"), llm_mode="TEST MODE", top_k=3,
        sandbox=fake_sandbox(timeout_seconds=120, max_output_bytes=60000), max_repairs=1,
        memory=MemoryStore(tmp_path / "mem.jsonl"), before_execute=hook)


def test_spec_mode_pipeline_runs_a_real_experiment_and_remembers_it(workflow_parts, tmp_path):
    pytest.importorskip("sklearn")
    calls = []
    wf = spec_workflow(workflow_parts, tmp_path, hook=lambda: calls.append("unloaded"))
    result = wf.run("graph neural networks")
    assert result.status == "COMPLETED" and result.execution_status == Status.SUCCESS
    assert result.experiment.source == "model_spec" and result.experiment.design_summary.startswith("iris:")
    assert result.critic.verdict in ("SIGNIFICANT", "INCONCLUSIVE") and result.critic.data_origin == "benchmark"
    assert calls == ["unloaded"]                                        # the model was unloaded before the sandbox started
    assert set(result.step_seconds) == {"retrieve", "review", "hypothesis", "experiment_design", "execute", "critic"}
    saved = wf.memory.entries()
    assert len(saved) == 1 and saved[0]["design"] == result.experiment.design_summary
    wf.run("graph neural networks")                                     # the second run is shown the first
    assert "Previous attempts" in wf.memory.render("graph neural networks", include_mock=True)
    assert len(wf.memory.entries()) == 2


def test_a_failing_unload_hook_does_not_stop_the_run(workflow_parts, tmp_path):
    pytest.importorskip("sklearn")
    def boom():
        raise RuntimeError("cannot unload")
    result = spec_workflow(workflow_parts, tmp_path, hook=boom).run("graph neural networks")
    assert result.execution_status == Status.SUCCESS


def test_invalid_designs_fall_back_to_a_labelled_default(workflow_parts, tmp_path):
    pytest.importorskip("sklearn")
    executor = CodeExecutorAgent(FakeListChatModel(responses=["no json here"]), mode="spec", spec_retries=0)
    result = spec_workflow(workflow_parts, tmp_path, executor=executor).run("graph neural networks")
    assert result.experiment.source == "default_spec"
    assert any("does NOT test the hypothesis" in w for w in result.warnings)
    assert result.execution_status == Status.SUCCESS


def test_settings_for_experiment_mode_are_validated():
    s = load_settings(env={}, load_env_file=False)
    assert (s.code_mode, s.spec_retries, s.memory_char_budget, s.unload_model_before_run, s.figure_vlm_model) == ("spec", 2, 1200, True, "")
    assert load_settings(env={"UNLOAD_MODEL_BEFORE_RUN": "false"}, load_env_file=False).unload_model_before_run is False
    for bad in ({"CODE_MODE": "magic"}, {"SPEC_RETRIES": "9"}, {"MEMORY_CHAR_BUDGET": "-1"}):
        with pytest.raises(ConfigError):
            load_settings(env=bad, load_env_file=False)
