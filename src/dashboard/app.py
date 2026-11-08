"""Streamlit monitoring dashboard. Start it with:  python -m src.main dashboard"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # so `import src...` works under `streamlit run`

import streamlit as st

from config.settings import load_settings
from src.dashboard.data import citations, cost_summary, load_runs, overview_rows, trace_steps
from src.draft.generator import write_draft

st.set_page_config(page_title="Discovery agent monitor", layout="wide")
st.title("Discovery agent monitor")
st.caption("Read-only view of saved runs for human supervisors. Results are small checks on toy datasets, not verified "
           "research findings.")

settings = load_settings()
entries = load_runs(settings.runs_dir)
if not entries:
    st.info("No run records yet. Run: python -m src.main run --topic \"...\"")
    st.stop()

only_real = st.sidebar.checkbox("Only runs made with a real model", value=True)
shown = [e for e in entries if not (only_real and e.is_test_mode)] or entries
index = st.sidebar.selectbox("Run", range(len(shown)), format_func=lambda i: shown[i].label)
entry = shown[index]
record = entry.record
st.sidebar.write(f"File: {entry.path.name}")
if entry.is_test_mode:
    st.warning("TEST MODE: this run used the mock model. Its text is a placeholder and its results are not real.")

st.subheader(f"Topic: {record.get('topic')}")
tab_tree, tab_cite, tab_cost, tab_all, tab_draft = st.tabs(
    ["Logic tree", "Literature citations", "Execution costs", "All runs", "Draft"])

with tab_tree:
    st.write("The agent's steps in order. Steps with a warning or a failure are opened automatically.")
    for number, step in enumerate(trace_steps(record), 1):
        took = f"{step.seconds} s" if step.seconds is not None else "time not recorded"
        with st.expander(f"{number}. {step.title}: {step.status.upper()} ({took})", expanded=step.status in ("warning", "failed")):
            for line in step.lines or ["(nothing recorded)"]:
                st.write(line)

with tab_cite:
    cites = citations(record)
    if cites["unknown"]:
        st.error("Cited but never retrieved (possible invented citations): " + ", ".join(cites["unknown"]))
    st.dataframe(cites["rows"], use_container_width=True,
                 column_config={"link": st.column_config.LinkColumn("link")})
    st.caption("A paper is only counted as cited when its id appears in the answer. Similarity is the search score.")

with tab_cost:
    cost = cost_summary(record)
    a, b, c, d = st.columns(4)
    a.metric("Total time (s)", cost["total_seconds"])
    b.metric("Sandbox time (s)", cost["sandbox_seconds"])
    c.metric("Sandbox attempts", cost["attempts"])
    d.metric("Money", cost["money"])
    if cost["step_seconds"]:
        st.bar_chart(cost["step_seconds"])
    st.caption(f"Model: {cost['model']}. {cost['note']}")

with tab_all:
    st.dataframe(overview_rows(shown), use_container_width=True)

with tab_draft:
    st.write("Write a draft (or a failed run report) from this run. It is filled from the record without a language model.")
    if st.button("Write draft"):
        kind, files = write_draft(entry.path, settings.storage_dir / "drafts", "both")
        st.success(("Draft" if kind == "paper" else "Failed run report") + " written: " + ", ".join(f.name for f in files))
        st.markdown(files[0].read_text(encoding="utf-8"))
        for f in files:
            st.download_button(f"Download {f.name}", f.read_text(encoding="utf-8"), file_name=f.name)
