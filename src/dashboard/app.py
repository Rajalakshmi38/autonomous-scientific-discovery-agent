"""Streamlit dashboard. Start it with:  python -m src.main dashboard

Start research (it calls src.main.run_research, the same function as the terminal command), watch the real progress,
browse saved runs, delete runs and write drafts. Selecting a run, switching tabs or deleting never starts research:
only the Start Research button does, and only once per click.
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # so `import src...` works under `streamlit run`

import streamlit as st

from config.settings import load_settings
from src.dashboard.data import STEPS, citations, cost_summary, load_runs, overview_rows, trace_steps
from src.dashboard.manage import delete_all_runs, delete_run
from src.dashboard.runner import MAX_TOPIC_CHARS, fallback_queries, format_event, friendly_error, validate_topic, waiting_line
from src.dashboard.insights import RatingError, bandit_rows, evaluation_folders, load_evaluation, reset_bandit, save_ratings
from src.draft.generator import write_draft

st.set_page_config(page_title="Discovery agent monitor", layout="wide")
st.title("Discovery agent monitor")
st.caption("An autonomous AI research platform that retrieves scientific literature, generates hypotheses, designs "
           "experiments and evaluates research results.")

settings = load_settings()
RUNS_DIR, DRAFTS_DIR = settings.runs_dir, settings.storage_dir / "drafts"
ss = st.session_state
PER_RUN_PAGES = ["Logic tree", "Literature Citations", "Execution Performance & Costs", "Draft"]      # pages about ONE run
SECTIONS = ["Start New Research", "Runs"] + PER_RUN_PAGES + ["Evaluation", "Learning (bandit)"]
MENU = [("Research", [("Start New Research", ":material/add_circle:"), ("Runs", ":material/list_alt:")]),
        ("Selected run", [("Logic tree", ":material/account_tree:"), ("Literature Citations", ":material/menu_book:"),
                          ("Execution Performance & Costs", ":material/speed:"), ("Draft", ":material/description:")]),
        ("Project", [("Evaluation", ":material/fact_check:"), ("Learning (bandit)", ":material/psychology:")])]
ss["_dirs"] = (str(RUNS_DIR), str(DRAFTS_DIR))                    # the button callbacks below read these
ss["_storage_dir"] = str(settings.storage_dir)
busy_eval = bool(ss.get("pending_eval"))                          # an evaluation is about to start in this very run of the script
busy = bool(ss.get("pending_topic")) or busy_eval                 # a research run is about to start in this very run of the script
if busy:
    ss["section"] = "Evaluation" if busy_eval else SECTIONS[0]    # a run starts from (and is shown on) its own page


# ------------------------------------------------------------------ callbacks
# Callbacks run exactly once per click, before the page reruns. State changes (and deletions) happen here, never in the
# page body, so an ordinary rerun can not repeat them.
def _flash(level: str, text: str) -> None:
    ss.setdefault("flash", []).append((level, text))


def _dirs():
    runs, drafts = ss["_dirs"]
    return Path(runs), Path(drafts)


def _on_start() -> None:
    if ss.get("pending_topic"):
        _flash("warning", "A research run is already starting.")
        return
    check = validate_topic(ss.get("topic_input", ""))
    if not check.ok:
        _flash("error", check.error)
        return
    options = {                                                   # the same choices as the terminal flags
        "k": int(ss.get("opt_k") or 5),                           # run -k
        "adaptive": bool(ss.get("opt_adaptive")),                 # run --adaptive (the bandit chooses k)
        "fetch": bool(ss.get("opt_fetch")),                       # ingest first
        "source": ss.get("opt_source") or "arxiv",
        "fetch_n": int(ss.get("opt_fetch_n") or 30),              # ingest --max-results
        "category": " ".join(str(ss.get("opt_category") or "").split()) or None,   # ingest --category (arXiv only)
        "keywords": " ".join(str(ss.get("opt_keywords") or "").split()) or None,   # ingest --query (default: the topic)
    }
    if options["keywords"] and len(options["keywords"]) > MAX_TOPIC_CHARS:
        _flash("error", f"The search keywords are too long. Keep them under {MAX_TOPIC_CHARS} characters.")
        return
    ss["confirm_delete"], ss["confirm_clear"] = None, False
    ss["pending_options"] = options
    ss["pending_topic"] = check.topic                             # the script body picks this up once and clears it



def _goto(page: str) -> None:
    ss["section"] = page
    ss["confirm_delete"], ss["confirm_clear"], ss["confirm_bandit_reset"] = None, False, False   # no stale question on the next page


def _open_run(name: str) -> None:
    ss["current_run"], ss["section"] = name, "Logic tree"
    ss["_runs_ver"] = ss.get("_runs_ver", 0) + 1                  # a new table key: no row stays selected


def _picked() -> None:
    ss["current_run"] = ss.get("selected_run")                    # the run dropdown on the run pages


def _toggle_real() -> None:
    ss["_only_real"] = bool(ss.get("only_real_cb", True))


def _ask_delete(name: str) -> None:
    ss["confirm_delete"], ss["confirm_clear"] = name, False


def _ask_clear() -> None:
    ss["confirm_delete"], ss["confirm_clear"] = None, True


def _cancel() -> None:
    ss["confirm_delete"], ss["confirm_clear"] = None, False


def _do_delete() -> None:
    name, ss["confirm_delete"] = ss.get("confirm_delete"), None   # clear first: one click deletes at most once
    if not name:
        return
    runs, drafts = _dirs()
    result = delete_run(runs, drafts, name)
    if result.errors:
        _flash("error", "Could not delete the run. " + " ".join(result.errors))
    elif result.missing and not result.deleted:
        _flash("info", f"{name} was already deleted.")
    else:
        extra = len(result.deleted) - 1
        _flash("success", f"Deleted run {name}" + (f" and {extra} draft file(s)." if extra > 0 else "."))
    ss["_runs_ver"] = ss.get("_runs_ver", 0) + 1                  # a new table key: the old row selection can not point at another run
    if ss.get("current_run") == name:
        ss.pop("current_run", None)                               # the page then selects another run, or shows the empty state
    ss.get("draft_cache", {}).pop(name, None)


def _do_clear() -> None:
    ss["confirm_clear"] = False
    runs, drafts = _dirs()
    result = delete_all_runs(runs, drafts)
    if result.errors:
        _flash("error", f"Clear All Runs finished with problems ({result.runs_deleted} run(s) deleted). " + " ".join(result.errors[:3]))
    else:
        _flash("success", f"Deleted all saved research runs ({result.runs_deleted}). Shared papers and the search index were kept.")
    ss.pop("current_run", None)
    ss["draft_cache"] = {}
    ss["_runs_ver"] = ss.get("_runs_ver", 0) + 1


def _dismiss_last_job() -> None:
    ss.pop("last_job", None)


def _start_eval(mode: str) -> None:
    """Preview or run the evaluation: the same choices as the flags of `python -m src.main evaluate`."""
    if ss.get("pending_eval") or ss.get("pending_topic"):
        _flash("warning", "A run is already starting.")
        return
    category = " ".join(str(ss.get("ev_category") or "").split())
    if not category:
        _flash("error", "Enter an arXiv category, for example cs.LG.")
        return
    ss["confirm_delete"], ss["confirm_clear"], ss["confirm_bandit_reset"] = None, False, False
    ss["pending_eval"] = {"mode": mode, "category": category, "n": int(ss.get("ev_n") or 5),
                          "pool": int(ss.get("ev_pool") or 30), "k": int(ss.get("ev_k") or 3)}


def _save_ratings() -> None:
    name = ss.get("eval_folder")
    folder = Path(ss["_storage_dir"]) / "evaluation" / str(name)
    if name not in [f.name for f in evaluation_folders(Path(ss["_storage_dir"]))]:
        _flash("error", "That evaluation no longer exists.")
        return
    try:
        rows = load_evaluation(folder)["rows"]
        scores = []
        for i in range(len(rows)):
            key = f"rate_{name}_{i}_"
            scores.append({"relevance": ss.get(key + "rel"), "testability": ss.get(key + "test"),
                           "groundedness": ss.get(key + "ground"), "comments": ss.get(key + "com")})
        changed = save_ratings(folder / "rating_sheet.csv", scores, ss.get("rate_rater", ""))
    except (RatingError, OSError, ValueError) as exc:
        _flash("error", f"The ratings were not saved. {exc}")
        return
    _flash("success" if changed else "info",
           f"Saved the ratings of {changed} paper(s). The previous sheet was kept as rating_sheet.previous.csv." if changed
           else "Nothing changed, so nothing was saved.")


def _ask_bandit_reset() -> None:
    ss["confirm_bandit_reset"], ss["confirm_delete"], ss["confirm_clear"] = True, None, False


def _cancel_bandit_reset() -> None:
    ss["confirm_bandit_reset"] = False


def _do_bandit_reset() -> None:
    ss["confirm_bandit_reset"] = False                            # clear first: one click resets at most once
    try:
        had = reset_bandit(Path(ss["_storage_dir"]) / "bandit.json")
    except OSError as exc:
        _flash("error", f"The bandit could not be reset: {exc.strerror or exc}")
        return
    _flash("success" if had else "info", "The bandit was reset. A copy was kept as bandit.json.bak." if had else "There was nothing to reset.")


def _go_adaptive() -> None:
    ss["section"], ss["opt_adaptive"] = SECTIONS[0], True         # open the Start page with the bandit option ticked


@st.cache_resource
def _run_lock() -> threading.Lock:
    """One research run at a time per dashboard server (the local model and Docker are single-user resources)."""
    return threading.Lock()


# ------------------------------------------------------------------ helpers


def _when(record, path) -> str:
    return str(record.get("created_at", path.stem)).replace("T", " ")[:16]


def _describe_model() -> str:
    try:
        from src.llm.factory import describe_llm
        return describe_llm(settings)
    except Exception:                                             # a label only; never block the page
        return settings.llm_provider


def _fetch_papers(topic: str, options: dict, slot):
    """The optional first stage: the same code as `python -m src.main ingest`. Returns (error message or None, line).
    Without your own keywords it tries the whole topic first and then its main words, because arXiv often finds nothing for a sentence."""
    from src.main import ingest_papers
    source = options["source"]
    category = options["category"] if source == "arxiv" else None
    queries = [options["keywords"]] if options["keywords"] else fallback_queries(topic)
    for query in queries:
        slot.markdown(f":blue[:material/hourglass_top:] Fetching up to {options['fetch_n']} papers from {source} for \"{query}\"...")
        try:
            papers = ingest_papers(settings, query, source=source, max_results=options["fetch_n"], category=category)
        except Exception as exc:                                  # network, source or index problem
            return friendly_error(exc, "Fetch papers"), None
        if papers:
            line = f":green[:material/check_circle:] Fetched, stored and indexed {len(papers)} paper(s) from {source} using \"{query}\"."
            slot.markdown(line)
            return None, line
    tried = ", ".join(f'"{q}"' for q in queries)
    line = f":orange[:material/warning:] No papers were returned for {tried}, so the run continues with the papers already in the index."
    slot.markdown(line)
    return None, line


def _run_job(topic: str, options: dict, area) -> None:
    """Run the existing pipeline for one topic, showing each step as it really starts and ends."""
    from src.main import llm_problem, run_research                 # imported here so the page opens fast
    from src.workflow import PIPELINE_STEPS
    from src.utils.logging_config import setup_logging

    lock = _run_lock()
    if not lock.acquire(blocking=False):
        _flash("warning", "Another research run is already in progress. Wait for it to finish, then start yours.")
        return
    lines = {key: waiting_line(i, len(PIPELINE_STEPS), title) for i, (key, title) in enumerate(PIPELINE_STEPS, 1)}
    job = {"topic": topic, "ok": False, "lines": [], "run": None}
    ss["running_topic"] = topic                                   # only cleared when this function ends normally (see the interrupted notice)
    current = {"key": None}
    try:
        with area:
            with st.status(f"Research in progress: {topic}", expanded=True) as status:
                st.caption(f"Model: {_describe_model()}. Leaving or using the page while this runs stops it and nothing is saved.")
                fetch_slot = st.empty() if options["fetch"] else None      # the fetch line sits ABOVE the six steps
                slots = {key: st.empty() for key, _ in PIPELINE_STEPS}
                for key, _ in PIPELINE_STEPS:
                    slots[key].markdown(lines[key])
                problem = llm_problem(settings)
                fetch_error = None
                if not problem and options["fetch"]:
                    fetch_error, fetch_line = _fetch_papers(topic, options, fetch_slot)
                    if fetch_line:
                        lines = {"_fetch": fetch_line, **lines}   # shown before the six steps
                if problem:
                    job["lines"] = [f":red[:material/error:] Cannot start: {problem}"]
                    status.update(label="Cannot start research", state="error", expanded=True)
                    _flash("error", f"Cannot start research: {problem}")
                elif fetch_error:
                    job["lines"] = [f":red[:material/error:] {fetch_error}"]
                    status.update(label="Fetching papers failed. The research was not started.", state="error", expanded=True)
                    _flash("error", fetch_error)
                else:
                    def on_progress(event) -> None:                # called by the workflow as each step really starts and ends
                        current["key"] = event.step
                        lines[event.step] = format_event(event)
                        slots[event.step].markdown(lines[event.step])
                    try:
                        setup_logging(settings.log_level, settings.logs_dir)       # same log file as the terminal command
                        def note(text: str) -> None:           # e.g. which paper count the bandit chose
                            lines[f"_note{len(lines)}"] = f":blue[:material/info:] {text}"
                            st.write(text)
                        finished = run_research(settings, topic, k=options["k"], adaptive=options["adaptive"],
                                                on_progress=on_progress, notify=note)
                    except Exception as exc:                       # an agent, the model or the storage failed
                        step_title = dict(PIPELINE_STEPS).get(current["key"])
                        message = friendly_error(exc, step_title)
                        lines["_error"] = f":red[:material/error:] {message}"
                        job["lines"] = [lines[k] for k in lines]
                        status.update(label="Research stopped by an error. Nothing was saved.", state="error", expanded=True)
                        _flash("error", message)
                    else:
                        result = finished.result
                        job.update(ok=True, run=finished.path.name)
                        status.update(label="Research finished and saved", state="complete", expanded=False)
                        ss["_select_after"] = finished.path.name  # applied at the top of the next pass, before the sidebar widgets exist
                        ss["_section_after"] = "Logic tree"       # show the new run's steps and result straight away
                        if "TEST MODE" in str(result.llm_mode):
                            ss["_show_mock_after"] = True         # a mock-model run would otherwise be hidden by the filter
                        if result.status == "NO_EVIDENCE":
                            _flash("warning", "The run was saved, but no relevant papers were found in the local index, so no later "
                                              "step ran. Add papers first: python -m src.main ingest --query \"...\"")
                        else:
                            _flash("success", f"Research finished. Run saved as {finished.path.name}. Execution: {result.execution_status}.")
                        if options["adaptive"]:
                            reward = f" and recorded a reward of {finished.reward:.2f}" if finished.reward is not None else ""
                            _flash("info", f"The bandit chose to retrieve {finished.k} paper(s){reward}.")
                        for w in result.warnings[:6]:
                            _flash("warning", str(w))
                        job["lines"] = [lines[k] for k in lines]
        ss["last_job"] = job
        ss["running_topic"] = None
    finally:
        lock.release()


def _run_eval_job(options: dict, area) -> None:
    """Preview or run the holdout evaluation, using the same functions as `python -m src.main evaluate`."""
    from src.main import llm_problem, pick_evaluation_papers, run_evaluation
    from src.utils.logging_config import setup_logging

    lock = _run_lock()
    if not lock.acquire(blocking=False):
        _flash("warning", "Another run is already in progress. Wait for it to finish, then try again.")
        return
    running = options["mode"] == "run"
    ss["running_eval"] = options["mode"]                          # only cleared when this function ends normally
    try:
        with area:
            with st.status("Evaluation in progress" if running else "Looking for recent papers", expanded=True) as status:
                try:
                    setup_logging(settings.log_level, settings.logs_dir)
                    st.write(f"Looking for recent arXiv papers in {options['category']} that are newer than everything in the index...")
                    chosen, cutoff = pick_evaluation_papers(settings, category=options["category"], n=options["n"], pool=options["pool"])
                except Exception as exc:                          # empty index, no internet, arXiv problem
                    status.update(label="Could not choose papers", state="error", expanded=True)
                    _flash("error", str(exc) if isinstance(exc, ValueError) else friendly_error(exc, "Choose held-out papers"))
                    ss["running_eval"] = None
                    return
                if not chosen:
                    status.update(label="No suitable paper found", state="complete", expanded=False)
                    _flash("warning", "No suitable recent paper was found. Look through more papers or try another category.")
                    ss["running_eval"] = None
                    return
                if not running:
                    ss["eval_preview"] = {"cutoff": cutoff, "papers": [{"paper": p.uid, "published": p.published[:10], "title": p.title} for p in chosen]}
                    status.update(label="Preview ready. Nothing was run.", state="complete", expanded=False)
                    ss["running_eval"] = None
                    return
                problem = llm_problem(settings)
                if problem:
                    status.update(label="Cannot start the evaluation", state="error", expanded=True)
                    _flash("error", f"Cannot start the evaluation: {problem}")
                    ss["running_eval"] = None
                    return
                ss.pop("eval_preview", None)
                st.caption("Each paper takes several minutes. Ollama and Docker must stay running, and using the page stops the evaluation.")
                total = len(chosen)
                slots = [st.empty() for _ in chosen]
                for i, p in enumerate(chosen):
                    slots[i].markdown(f":gray[:material/radio_button_unchecked:] Paper {i + 1} of {total}: {p.title[:90]} - waiting")
                step_slot = st.empty()

                def on_paper(i: int, count: int, title: str) -> None:
                    if i > 1:
                        slots[i - 2].markdown(f":green[:material/check_circle:] Paper {i - 1} of {total}: {chosen[i - 2].title[:90]} - processed")
                    if i >= 1:
                        slots[i - 1].markdown(f":blue[:material/hourglass_top:] Paper {i} of {total}: {title[:90]} - running...")

                def on_progress(event) -> None:
                    step_slot.markdown(format_event(event))
                try:
                    folder, outcome = run_evaluation(settings, chosen, cutoff, k=options["k"], on_progress=on_progress, on_paper=on_paper)
                except Exception as exc:
                    status.update(label="The evaluation stopped by an error", state="error", expanded=True)
                    _flash("error", friendly_error(exc, None) + " Runs that finished before the error are saved under Runs; no evaluation folder was written.")
                else:
                    slots[-1].markdown(f":green[:material/check_circle:] Paper {total} of {total}: {chosen[-1].title[:90]} - processed")
                    step_slot.empty()
                    status.update(label="Evaluation finished and saved", state="complete", expanded=False)
                    summary = outcome["summary"]
                    ss["_eval_after"] = folder.name
                    _flash("success", f"Evaluation finished: {summary['runs_finished']} of {summary['papers']} run(s) completed. Results are in {folder.name}. "
                                      "Rate the papers below.")
        ss["running_eval"] = None
    finally:
        lock.release()


# ------------------------------------------------------------------ messages, interrupted run, confirmations
if "_select_after" in ss:                                          # a run just finished: select it (widgets are not created yet)
    ss["current_run"] = ss.pop("_select_after")
if "_eval_after" in ss:                                            # an evaluation just finished: show its results
    ss["eval_folder"] = ss.pop("_eval_after")
if "_section_after" in ss:
    ss["section"] = ss.pop("_section_after")
if ss.pop("_show_mock_after", False):
    ss["_only_real"] = False
entries = load_runs(RUNS_DIR)
unreadable = len(list(RUNS_DIR.glob("run_*.json"))) - len(entries) if RUNS_DIR.exists() else 0

interrupted = ss.pop("running_topic", None)
if interrupted:
    st.warning(f"The research run for \"{interrupted}\" did not finish: the page was reloaded or used while it was running. "
               "Nothing was saved for it. Start it again when you are ready.")

if ss.pop("running_eval", None):
    st.warning("The evaluation did not finish: the page was reloaded or used while it was running. Runs that finished before that are saved "
               "under Runs, but no evaluation folder or rating sheet was written for it. Start it again when you are ready.")

for level, text in ss.pop("flash", []):
    getattr(st, level, st.info)(text)

if not busy and ss.get("confirm_delete"):
    target = next((e for e in entries if e.path.name == ss["confirm_delete"]), None)
    with st.container(border=True):
        st.warning(f"**Delete this research run?**\n\n{(target.record.get('topic') if target else ss['confirm_delete'])} "
                   f"({ss['confirm_delete']})\n\nIts saved results and generated drafts will be permanently deleted. "
                   "Shared papers, the search index and the agent memory are not affected.")
        a, b, _ = st.columns([1, 1, 4])
        a.button("Delete run", key="confirm_delete_yes", type="primary", icon=":material/delete:", on_click=_do_delete)
        b.button("Cancel", key="confirm_delete_no", on_click=_cancel)

if not busy and ss.get("confirm_clear"):
    hidden_test = sum(1 for e in entries if e.is_test_mode)
    with st.container(border=True):
        st.warning(f"**Clear all saved research runs?**\n\nThis will permanently delete all {len(entries)} saved research run(s)"
                   + (f", including {hidden_test} made with the mock model" if hidden_test else "")
                   + (f", and {unreadable} unreadable run file(s)" if unreadable > 0 else "")
                   + " together with their generated drafts. Shared papers, the search index and the agent memory are kept. "
                     "This cannot be undone.")
        a, b, _ = st.columns([2, 1, 3])
        a.button("Yes, delete all runs", key="confirm_clear_yes", type="primary", icon=":material/delete_forever:", on_click=_do_clear)
        b.button("Cancel", key="confirm_clear_no", on_click=_cancel)

# ------------------------------------------------------------------ which run is selected
ss.setdefault("_only_real", True)
shown = [e for e in entries if not (ss["_only_real"] and e.is_test_mode)] or entries
by_name = {e.path.name: e for e in shown}
if shown and ss.get("current_run") not in by_name:
    ss["current_run"] = shown[0].path.name
entry = by_name.get(ss.get("current_run")) if shown else None

# ------------------------------------------------------------------ sidebar: the menu
if ss.get("section") not in SECTIONS:
    ss["section"] = SECTIONS[0]
with st.sidebar:
    st.markdown("<style>[data-testid='stSidebar'] .stButton > button{justify-content:flex-start;text-align:left}</style>",
                unsafe_allow_html=True)                           # menu items read from the left, like a normal menu
    for group, items in MENU:
        st.caption(group)
        for label, icon in items:
            st.button(label, key=f"nav_{SECTIONS.index(label)}", icon=icon, width="stretch", disabled=busy,
                      type="primary" if ss["section"] == label else "tertiary", on_click=_goto, args=(label,))

# ------------------------------------------------------------------ start new research
section = ss["section"]
if section == SECTIONS[0]:
    with st.container(border=True):
        st.subheader("Start New Research")
        with st.form("start_research", border=False):
            st.text_input("Research topic", key="topic_input", max_chars=MAX_TOPIC_CHARS, disabled=busy,
                          placeholder="e.g. Diabetes Prediction Using Machine Learning")
            k_col, bandit_col = st.columns([1, 2], vertical_alignment="bottom")
            k_col.number_input("Papers to retrieve (k)", min_value=1, max_value=10, value=5, step=1, key="opt_k", disabled=busy,
                               help="How many of the closest papers the agents read. Ignored when the bandit chooses.")
            bandit_col.checkbox("Let the bandit (reinforcement learning) choose how many papers to retrieve", key="opt_adaptive",
                                disabled=busy, help="Same as --adaptive: it picks 2, 3 or 5 papers and learns from the result.")
            st.checkbox("Fetch new papers for this topic first (same as the ingest command)", key="opt_fetch", disabled=busy,
                        help="Downloads papers into your local index before the run. Leave it off to use the papers already stored.")
            src_col, n_col, cat_col = st.columns(3)
            src_col.selectbox("Source (used when fetching)", ["arxiv", "pubmed"], key="opt_source", disabled=busy)
            n_col.number_input("Papers to fetch", min_value=1, max_value=100, value=30, step=5, key="opt_fetch_n", disabled=busy)
            cat_col.text_input("arXiv category (optional)", value="cs.LG", key="opt_category", disabled=busy,
                               help="arXiv only, for example cs.LG. Ignored for PubMed.")
            st.text_input("Search keywords for fetching (optional)", key="opt_keywords", disabled=busy,
                          placeholder="Leave empty to search with the topic itself")
            st.form_submit_button("Start Research", type="primary", icon=":material/play_arrow:", on_click=_on_start, disabled=busy)
        job_area = st.container()
        last = ss.get("last_job")
        if last and not busy:
            with job_area:
                with st.status(f"Last research run: {last['topic']}", state="complete" if last["ok"] else "error", expanded=not last["ok"]):
                    for line in last["lines"]:
                        st.markdown(line)
                    st.button("Dismiss", key="dismiss_last_job", on_click=_dismiss_last_job)

    if busy:
        topic_to_run = ss.pop("pending_topic")                         # cleared BEFORE running: no rerun can start it twice
        run_options = ss.pop("pending_options", None) or {"k": 5, "adaptive": False, "fetch": False, "source": "arxiv",
                                                          "fetch_n": 30, "category": None, "keywords": None}
        ss.pop("last_job", None)
        _run_job(topic_to_run, run_options, job_area)
        st.rerun()                                                    # refresh the run list and the selected run
    if not entries:
        st.info("No run records yet. Enter a research topic above and press Start Research.")
    st.stop()

# ------------------------------------------------------------------ the other sections
if section == "Evaluation":
    st.subheader("Evaluation on recent unseen papers")
    folder, ev = None, None
    try:
        folders = evaluation_folders(settings.storage_dir)
        if not folders:
            st.info("No evaluation has been run yet. Use the form at the bottom of this page, or: python -m src.main evaluate")
        else:
            names = [f.name for f in folders]
            if ss.get("eval_folder") not in names:
                ss["eval_folder"] = names[0]
            st.selectbox("Evaluation", names, key="eval_folder", disabled=busy)
            folder = settings.storage_dir / "evaluation" / ss["eval_folder"]
            ev = load_evaluation(folder)
            if ev["summary"]:
                st.markdown(ev["summary"])
            if ev["rows"]:
                st.markdown("**Each held-out paper**")
                st.dataframe(ev["rows"], hide_index=True, width="stretch",
                             column_order=["paper", "cites", "execution", "verdict", "similarity", "rank", "ratings", "rated_by"],
                             column_config={
                                 "paper": st.column_config.TextColumn("Held-out paper", width="large"),
                                 "cites": st.column_config.TextColumn("Cites a retrieved paper"),
                                 "execution": st.column_config.TextColumn("Execution"),
                                 "verdict": st.column_config.TextColumn("Verdict"),
                                 "similarity": st.column_config.NumberColumn("Similarity to its own paper", format="%.3f"),
                                 "rank": st.column_config.NumberColumn("Own paper rank"),
                                 "ratings": st.column_config.TextColumn("Ratings (1 to 5)", width="medium"),
                                 "rated_by": st.column_config.TextColumn("Rated by", width="medium"),
                             })
                st.caption(f"Ratings filled in: {ev['rated']} of {len(ev['rows'])} paper(s). Check the \"Rated by\" column: a rating made by an AI assistant "
                           "is not a human rating. The similarity numbers are a rough proxy, not a quality score.")
    except Exception as exc:                                            # a damaged file must not break the page
        st.error(f"The evaluation files could not be read: {exc}")
        folder, ev = None, None

    if folder is not None and ev and ev["rows"]:
        st.markdown("**Rate the papers**")
        st.caption("Rate each hypothesis from 1 (poor) to 5 (good) for relevance to its paper, testability and groundedness in the cited papers. "
                   "Type your own name as the rater. The previous sheet is kept as rating_sheet.previous.csv.")
        with st.form("rate_form"):
            for i, row in enumerate(ev["rows"]):
                st.markdown(f"**{i + 1}. {row['paper']}**")
                key = f"rate_{folder.name}_{i}_"
                c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
                c1.number_input("Relevance", 1, 5, value=row["relevance"], step=1, key=key + "rel", placeholder="1 to 5", disabled=busy)
                c2.number_input("Testability", 1, 5, value=row["testability"], step=1, key=key + "test", placeholder="1 to 5", disabled=busy)
                c3.number_input("Groundedness", 1, 5, value=row["groundedness"], step=1, key=key + "ground", placeholder="1 to 5", disabled=busy)
                c4.text_input("Comment (optional)", value=row["comments"], key=key + "com", disabled=busy)
            st.text_input("Rater (your name)", key="rate_rater", placeholder="Required when you change a rating", disabled=busy)
            st.form_submit_button("Save ratings", icon=":material/save:", on_click=_save_ratings, disabled=busy)

    st.markdown("**Run a new evaluation**")
    st.caption("Looks for recent arXiv papers that are newer than everything in your index, then runs the full agent on each title. "
               "It needs the internet, Ollama and Docker, and each paper takes several minutes. Preview first if you are unsure.")
    with st.form("eval_form", border=False):
        f1, f2, f3, f4 = st.columns(4)
        f1.text_input("arXiv category", value="cs.LG", key="ev_category", disabled=busy)
        f2.number_input("Held-out papers", 1, 10, value=5, step=1, key="ev_n", disabled=busy)
        f3.number_input("Papers to look through", 10, 100, value=30, step=5, key="ev_pool", disabled=busy)
        f4.number_input("Papers retrieved per run (k)", 1, 10, value=3, step=1, key="ev_k", disabled=busy)
        g1, g2, _ = st.columns([1, 1, 3])
        g1.form_submit_button("Preview papers", on_click=_start_eval, args=("preview",), disabled=busy)
        g2.form_submit_button("Run evaluation", type="primary", icon=":material/play_arrow:", on_click=_start_eval, args=("run",), disabled=busy)
    eval_area = st.container()
    preview = ss.get("eval_preview")
    if preview and not busy_eval:
        with eval_area:
            st.markdown(f"**Preview: {len(preview['papers'])} paper(s)** newer than {preview['cutoff']} and not in the index")
            st.dataframe(preview["papers"], hide_index=True, width="stretch")
            st.caption("Preview only. Nothing was run and nothing was added to the index.")
    if busy_eval:
        eval_options = ss.pop("pending_eval")                      # cleared BEFORE running: no rerun can start it twice
        _run_eval_job(eval_options, eval_area)
        st.rerun()
    st.stop()

if section == "Learning (bandit)":
    st.subheader("What the reinforcement learning has learned")
    st.caption("The bandit chooses how many papers to retrieve (2, 3 or 5) and is rewarded by automatic checks on each saved run. "
               "It learns only from runs started with the bandit option.")
    if ss.get("confirm_bandit_reset") and not busy:
        with st.container(border=True):
            st.warning("**Reset what the bandit has learned?**\n\nIt will forget every learning event and start again. A copy of the current "
                       "file is kept as bandit.json.bak. Saved runs are not affected.")
            ya, yb, _ = st.columns([1, 1, 4])
            ya.button("Yes, reset", key="bandit_reset_yes", type="primary", icon=":material/restart_alt:", on_click=_do_bandit_reset)
            yb.button("Cancel", key="bandit_reset_no", on_click=_cancel_bandit_reset)
    try:
        info = bandit_rows(settings.storage_dir / "bandit.json", settings.bandit_epsilon)
        if info["total"] == 0:
            st.info("The bandit has not learned anything yet. Start a run with the bandit option ticked.")
        else:
            st.dataframe(info["rows"], hide_index=True, width="stretch", column_config={
                "papers": st.column_config.NumberColumn("Papers retrieved"),
                "runs": st.column_config.NumberColumn("Times tried"),
                "average": st.column_config.TextColumn("Average reward (0 to 1)"),
            })
            st.metric("Learning events so far", info["total"])
            if info["best"] is not None:
                st.write(f"Best average so far: retrieving **{info['best']}** papers.")
            st.caption("The reward is a proxy for a clean run (the sandbox succeeded, the hypothesis cited a retrieved paper, the critic gave a verdict "
                       "without a design warning). It is not a measure of good science, and with so few runs this is not evidence that the strategy helps.")
    except Exception as exc:
        st.error(f"The bandit file could not be read: {exc}")
    st.markdown("**Actions**")
    act1, act2, _ = st.columns([2, 1, 3])
    act1.button("Run a research with the bandit", key="go_adaptive", icon=":material/play_arrow:", on_click=_go_adaptive, disabled=busy)
    act2.button("Reset learning", key="bandit_reset_ask", icon=":material/restart_alt:", on_click=_ask_bandit_reset,
                disabled=busy or not (settings.storage_dir / "bandit.json").exists())
    st.stop()

if entry is None:
    st.info("No run records yet. Open Start New Research in the sidebar and press Start Research.")
    st.stop()

record = entry.record
if section in PER_RUN_PAGES:
    ss["selected_run"] = ss["current_run"]                       # before the widget exists: the saved choice wins
    st.selectbox("Run", [e.path.name for e in shown], format_func=lambda n: by_name[n].label, key="selected_run", on_change=_picked)
    if entry.is_test_mode:
        st.warning("TEST MODE: this run used the mock model. Its text is a placeholder and its results are not real.")
    st.subheader(f"Topic: {record.get('topic')}")
    st.caption(f"{_when(record, entry.path)} · File: {entry.path.name} · Execution: {record.get('execution_status')} · "
               f"Verdict: {(record.get('critic') or {}).get('verdict') or 'no verdict'}")

if section == "Logic tree":
    st.write("The agent's steps in order. Steps with a warning or a failure are opened automatically.")
    for number, step in enumerate(trace_steps(record), 1):
        took = f"{step.seconds} s" if step.seconds is not None else "time not recorded"
        with st.expander(f"{number}. {step.title}: {step.status.upper()} ({took})", expanded=step.status in ("warning", "failed")):
            for line in step.lines or ["(nothing recorded)"]:
                st.write(line)

elif section == "Literature Citations":
    cites = citations(record)
    if cites["unknown"]:
        st.error("Cited but never retrieved (possible invented citations): " + ", ".join(cites["unknown"]))
    st.dataframe(
        cites["rows"], hide_index=True, width="stretch",
        column_order=["id", "title", "similarity", "status", "cited by review", "cited by hypothesis", "link"],
        column_config={
            "id": st.column_config.TextColumn("Paper ID", width="small"),
            "title": st.column_config.TextColumn("Paper title", width="large"),
            "similarity": st.column_config.NumberColumn("Similarity", format="%.3f", width="small",
                                                         help="The search score of the paper for this topic."),
            "status": st.column_config.TextColumn("Status", width="small"),
            "cited by review": st.column_config.CheckboxColumn("Cited by review", width="small"),
            "cited by hypothesis": st.column_config.CheckboxColumn("Cited by hypothesis", width="small"),
            "link": st.column_config.LinkColumn("Paper link", width="small", display_text=r"https?://(?:www\.)?([^/]+)/"),
        })
    st.caption("A paper is only counted as cited when its id appears in the answer. Similarity is the search score.")

elif section == "Execution Performance & Costs":
    cost = cost_summary(record)
    a, b, c, d = st.columns(4)
    a.metric("Total time (s)", cost["total_seconds"])
    b.metric("Sandbox time (s)", cost["sandbox_seconds"])
    c.metric("Sandbox attempts", cost["attempts"])
    more = (f" Current setup in .env (it may differ from the one this run used): Ollama model {settings.ollama_model}, "
            f"context {settings.ollama_num_ctx} tokens, up to {settings.llm_max_tokens} new tokens per call." if settings.llm_provider == "ollama" else "")
    d.metric("Money", cost["money"], help=f"Free local model. Model: {cost['model']}. {cost['note']}{more}")     # the details live in this tooltip
    if cost["step_seconds"]:
        import pandas as pd
        titles = dict(STEPS)
        order = [k for k, _ in STEPS if k in cost["step_seconds"]] + [k for k in cost["step_seconds"] if k not in titles]
        frame = pd.DataFrame({"Step": [titles.get(k, k) for k in order], "Seconds": [cost["step_seconds"][k] for k in order]})
        st.markdown("**Time per step (seconds)**")
        st.bar_chart(frame, x="Step", y="Seconds", horizontal=True, sort=False, x_label="", y_label="Seconds",
                     height=max(240, 52 * len(frame)))

elif section == "Runs":
    ss["only_real_cb"] = ss["_only_real"]
    st.checkbox("Only runs made with a real model", key="only_real_cb", on_change=_toggle_real)
    rows = overview_rows(shown)
    st.caption(f"{len(shown)} saved run(s). Tick the box at the start of a row, then open or delete that run."
               + (f" {len(entries) - len(shown)} run(s) made with the mock model are hidden by the filter." if len(shown) < len(entries) else "")
               + (f" {unreadable} unreadable run file(s) are not listed; Clear All Runs removes them too." if unreadable > 0 else ""))
    event = st.dataframe(
        rows, hide_index=True, width="stretch", height=min(440, 42 + 36 * len(rows)), key=f"runs_table_{ss.get('_runs_ver', 0)}",
        on_select="rerun", selection_mode="single-row",
        column_order=["time", "topic", "mode", "execution", "verdict", "seconds", "file"],
        column_config={
            "time": st.column_config.TextColumn("Time", width="medium"),
            "topic": st.column_config.TextColumn("Topic", width="large"),
            "mode": st.column_config.TextColumn("Mode", width="small"),
            "execution": st.column_config.TextColumn("Execution", width="medium"),
            "verdict": st.column_config.TextColumn("Verdict", width="medium"),
            "seconds": st.column_config.TextColumn("Seconds", width="small"),
            "file": st.column_config.TextColumn("File", width="medium"),
        })
    chosen = list(getattr(getattr(event, "selection", None), "rows", []) or [])
    picked = rows[chosen[0]] if chosen and 0 <= chosen[0] < len(rows) else None
    st.caption(f"Selected: {picked['topic']} ({picked['file']})" if picked else "No row selected.")
    b_open, b_del, b_clear, _ = st.columns([1, 1, 1.3, 3])
    b_open.button("Open", key="runs_open", icon=":material/open_in_new:", disabled=busy or picked is None, width="stretch",
                  on_click=_open_run, args=(picked["file"] if picked else "",))
    b_del.button("Delete", key="runs_delete", icon=":material/delete:", disabled=busy or picked is None, width="stretch",
                 on_click=_ask_delete, args=(picked["file"] if picked else "",))
    b_clear.button("Clear All Runs", key="clear_all_tab", icon=":material/delete_forever:", disabled=not entries, width="stretch",
                   on_click=_ask_clear)

elif section == "Draft":
    st.write("Write a draft (or a failed run report) from this run. It is filled from the record without a language model.")
    made = ss.setdefault("draft_cache", {})
    if st.button("Write draft", key="write_draft"):
        try:
            kind, files = write_draft(entry.path, DRAFTS_DIR, "both")
            made[entry.path.name] = (kind, [str(f) for f in files])
        except Exception as exc:                                    # an unreadable or incomplete record must not crash the page
            made.pop(entry.path.name, None)
            st.error(f"The draft could not be written: {exc}")
    if entry.path.name in made:
        kind, paths = made[entry.path.name]
        files = [Path(p) for p in paths]
        if not all(f.is_file() for f in files):
            made.pop(entry.path.name, None)
            st.warning("The draft files are no longer on disk. Press Write draft again.")
        else:
            st.success(("Draft" if kind == "paper" else "Failed run report") + " written: " + ", ".join(f.name for f in files))
            if kind != "paper":
                st.info("This run has no verified result, so a failed run report was written instead of a paper.")
            with st.container(border=True):
                st.markdown(files[0].read_text(encoding="utf-8"))
            md_col, tex_col, _ = st.columns([1, 1, 3])
            for col, f, label in zip((md_col, tex_col), files, ("Download Markdown (.md)", "Download LaTeX (.tex)")):
                col.download_button(label, f.read_text(encoding="utf-8"), file_name=f.name, key=f"dl_{f.name}", on_click="ignore")
