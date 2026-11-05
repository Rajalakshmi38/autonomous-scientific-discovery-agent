"""Memory tests. The entries below are HAND-MADE test data."""
import json

from src.memory.store import MemoryStore

REAL = "ollama:llama3.2:3b (local, free)"
MOCK = "TEST MODE (rule-based mock, no LLM reasoning)"


def add(store, topic, llm=REAL, **extra):
    store.add(topic=topic, hypothesis="h " + topic, design="iris: knn vs svm", execution_status="SUCCESS",
              critic_verdict="INCONCLUSIVE", llm_mode=llm, **extra)


def test_entries_round_trip_and_survive_damage(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    assert store.entries() == []
    add(store, "graph neural networks")
    with (tmp_path / "m.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{damaged line\n[1, 2]\n")
    assert [e["topic"] for e in store.entries()] == ["graph neural networks"]


def test_only_similar_topics_are_returned_newest_first(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    add(store, "graph neural networks node classification")
    add(store, "protein folding with transformers")
    add(store, "graph neural networks edge unlearning")
    found = store.similar("graph neural networks")
    assert [e["topic"] for e in found] == ["graph neural networks edge unlearning", "graph neural networks node classification"]


def test_mock_runs_never_steer_real_runs(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    add(store, "graph neural networks", llm=MOCK)
    assert store.similar("graph neural networks") == []
    assert len(store.similar("graph neural networks", include_mock=True)) == 1


def test_rendered_note_is_short_and_respects_the_budget(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    for i in range(30):
        add(store, f"graph neural networks variant {i}")
    note = store.render("graph neural networks", budget_chars=700)
    assert note.startswith("Previous attempts") and len(note) <= 700 + 300
    assert note.count("\n- ") < 6
    assert store.render("quantum chromodynamics lattice") == ""          # nothing relevant: no note at all


def test_the_file_is_trimmed(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl", max_entries=5)
    for i in range(12):
        add(store, f"graph neural networks run {i}")
    entries = store.entries()
    assert len(entries) == 5 and entries[-1]["topic"].endswith("11")
    assert all(json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines())


def test_memory_command(monkeypatch, tmp_path, capsys):
    from src.main import main
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "s"))
    assert main(["memory"]) == 0 and "Memory is empty" in capsys.readouterr().out
    add(MemoryStore(tmp_path / "s" / "memory.jsonl"), "graph neural networks")
    assert main(["memory"]) == 0
    assert "graph neural networks" in capsys.readouterr().out
