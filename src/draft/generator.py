"""Turn ONE saved run record into a draft (Markdown and LaTeX), without a language model.

Rules this module follows:
  * Every number comes from the run record. The critic's own summary lines are copied word for word.
  * Text written earlier by the language model (the literature summary, the hypothesis) is labelled as
    model-written and unverified.
  * A run that failed, timed out or could not be analysed gets a "failed run report", never a paper.
  * A run made with the mock model (TEST MODE) is labelled as such at the top.
  * Every draft says it is AI-assisted, unreviewed and not a verified research finding.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

Block = Tuple[str, Any]          # ("p", text) | ("bullets", [text]) | ("quote", text) | ("code", text)

PAPER_VERDICTS = {"SIGNIFICANT", "INCONCLUSIVE"}
DISCLAIMER = ("AI-assisted draft made from one saved run record. It has not been peer reviewed and its text is not a "
              "verified research finding.")
DESIGN_SOURCE_TEXT = {
    "model_spec": "The model chose the settings in a checked form, and a vetted template wrote the script.",
    "keyword_spec": "The model's own design was invalid, so fixed keyword rules chose the design from the hypothesis.",
    "default_spec": "The model's design was invalid, so a generic default design was used. It may not test the hypothesis.",
    "free_form": "The language model wrote the whole script itself.",
}
FAILED_ADVICE = {
    "TIMEOUT": "The script ran longer than the time limit. Raise SANDBOX_TIMEOUT_SECONDS in .env or choose a smaller design.",
    "SANDBOX_UNAVAILABLE": "The experiment was not run because Docker was not available. Start Docker Desktop and run again.",
    "OUT_OF_MEMORY": "The script used more memory than the sandbox allows. Choose a smaller dataset or method.",
    "FAILED": "The script raised an error. The last error text is shown above.",
    "NOT_RUN": "The experiment was not run (for example with --no-execute).",
}


@dataclass
class Draft:
    kind: str                                   # "paper" | "failed_run"
    title: str
    banners: List[str] = field(default_factory=list)
    abstract: str = ""
    sections: List[Tuple[str, List[Block]]] = field(default_factory=list)


# ------------------------------------------------------------------ reading records
def load_record(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not a valid run record (bad JSON: {exc})") from exc
    if not isinstance(data, dict) or "topic" not in data:
        raise ValueError(f"{path} does not look like a run record (no 'topic').")
    return data


def is_test_mode(record: Dict[str, Any]) -> bool:
    return "TEST MODE" in str(record.get("llm_mode", ""))


def latest_record_path(runs_dir: Path) -> Optional[Path]:
    """Newest run_*.json, preferring runs made with a real model."""
    files = sorted(Path(runs_dir).glob("run_*.json"), reverse=True) if Path(runs_dir).exists() else []
    for path in files:
        try:
            if not is_test_mode(json.loads(path.read_text(encoding="utf-8"))):
                return path
        except (OSError, json.JSONDecodeError):
            continue
    return files[0] if files else None


def draft_kind(record: Dict[str, Any]) -> str:
    critic = record.get("critic") or {}
    ok = (record.get("execution_status") == "SUCCESS" and critic.get("verdict") in PAPER_VERDICTS
          and bool(record.get("hypothesis")))
    return "paper" if ok else "failed_run"


# ------------------------------------------------------------------ building the draft
def _short(text: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _model_name(record: Dict[str, Any]) -> str:
    return str(record.get("llm_mode", "unknown model"))


def _banners(record: Dict[str, Any], kind: str) -> List[str]:
    out = [DISCLAIMER]
    if is_test_mode(record):
        out.insert(0, "TEST MODE: this record was made with the mock model. Its text is a placeholder and its "
                      "results are not real.")
    out.append(f"Generated from run record dated {record.get('created_at', 'unknown')} (model: {_model_name(record)}).")
    return out


def _evidence_blocks(record: Dict[str, Any]) -> List[Block]:
    cited = {s.get("id") for s in record.get("sources_cited_by_hypothesis") or []}
    lines = []
    for e in record.get("evidence") or []:
        mark = " (cited by the hypothesis)" if e.get("id") in cited else ""
        lines.append(f"[{e.get('id')}] {e.get('title')} (similarity {e.get('score')}){mark} {e.get('url', '')}".strip())
    return [("bullets", lines)] if lines else [("p", "No papers were retrieved.")]


def _spec_lines(spec: Optional[Dict[str, Any]]) -> List[str]:
    if not spec:
        return []

    def arm(a: Dict[str, Any]) -> str:
        params = ", ".join(f"{k}={v}" for k, v in (a.get("params") or {}).items()) or "default settings"
        return f"{a.get('model')} ({params}), feature scaling: {'yes' if a.get('scale') else 'no'}"
    return [f"Dataset: {spec.get('dataset')} (bundled with scikit-learn); metric: {spec.get('metric')}",
            f"Repeated random splits: {spec.get('n_resamples')}; test fraction: {spec.get('test_size')}; seed: {spec.get('seed')}",
            f"Baseline: {arm(spec.get('baseline') or {})}",
            f"Treatment: {arm(spec.get('treatment') or {})}"]


def _warning_bullets(record: Dict[str, Any]) -> List[str]:
    return [_short(w, 400) for w in record.get("warnings") or []]


def _limitations(record: Dict[str, Any]) -> List[Block]:
    critic = record.get("critic") or {}
    exp = record.get("experiment") or {}
    items = [f"[{f.get('severity')}] {f.get('message')}" for f in critic.get("findings") or []
             if f.get("severity") in ("warning", "info")]
    if exp.get("source") in ("keyword_spec", "default_spec"):
        items.append(DESIGN_SOURCE_TEXT[exp["source"]])
    if exp.get("note"):
        items.append(_short(exp["note"], 500))
    items += [f"Run warning: {w}" for w in _warning_bullets(record)]
    items += ["This is a single automated run. It was not repeated with other datasets, methods or seeds.",
              "The hypothesis and the literature summary were written by a small language model and can contain "
              "errors. Check them against the cited papers.",
              "The experiment may not test the hypothesis exactly. Compare the hypothesis with the design above."]
    return [("bullets", items)]


def _interpretation(verdict: str) -> str:
    if verdict == "SIGNIFICANT":
        return ("The critic found a statistically significant difference on this dataset. This alone does not make "
                "the hypothesis true, and it says nothing about other data.")
    return ("The critic found no significant difference. This does not show that there is no effect: the sample may be "
            "too small to detect one.")


def _paper(record: Dict[str, Any]) -> Draft:
    hyp, exp = record.get("hypothesis") or {}, record.get("experiment") or {}
    critic, execu = record.get("critic") or {}, record.get("execution") or {}
    verdict, origin = critic.get("verdict"), critic.get("data_origin", "unknown")
    lines = critic.get("summary_lines") or []
    design = exp.get("design_summary") or "(design not recorded)"
    abstract = (f"This draft reports one small automated experiment on the topic \"{record.get('topic')}\". "
                f"The hypothesis was: {hyp.get('statement')} The experiment compared: {design}. "
                f"The critic's verdict was {verdict} (data origin: {origin}). "
                + (f"Critic result: {lines[0].rstrip('. ')}. " if lines else "")
                + "The result describes only this dataset and is not a verified research finding.")
    review = (record.get("review") or {}).get("summary", "")
    intro: List[Block] = [("p", "Papers retrieved from the local index for this topic:"), *_evidence_blocks(record),
                          ("p", "Literature summary (written by the language model, not verified):"),
                          ("quote", _short(review, 3000) or "(none)"),
                          ("p", "Hypothesis (written by the language model): " + str(hyp.get("statement"))),
                          ("p", "Rationale (model-written): " + str(hyp.get("rationale", "")))]
    if hyp.get("test_plan"):
        intro.append(("p", "Test plan (model-written): " + str(hyp["test_plan"])))
    method: List[Block] = [("p", DESIGN_SOURCE_TEXT.get(exp.get("source"), "")),
                           ("p", f"Design: {design}")]
    if _spec_lines(exp.get("spec")):
        method.append(("bullets", _spec_lines(exp.get("spec"))))
    method.append(("p", f"The script ran in a Docker container with restricted resources and no network. Attempts: "
                        f"{execu.get('attempt_count', '?')}; final status: {execu.get('final_status', record.get('execution_status'))}. "
                        "The container restrictions were checked with the project's sandbox self-test, which is "
                        "evidence and not a guarantee."))
    method.append(("p", f"The critic recomputed all statistics from the raw scores (rule-based, no language model; "
                        f"significance level alpha = {critic.get('alpha')}). Repeated splits of the same data are not "
                        "independent, so resampled scores use the corrected resampled t-test."))
    results: List[Block] = [("p", f"Verdict: {verdict}. Data origin: {origin}."),
                            ("p", "Critic output, copied unchanged from the run record:"),
                            ("bullets", lines or ["(no summary lines recorded)"])]
    for comp in critic.get("comparisons") or []:
        res = comp.get("result")
        if res:
            names = comp.get("labels") or ["A", "B"]
            results.append(("p", f"{comp.get('name')}: mean of {names[0]} = {res.get('mean_a'):.4f}; "
                                 f"mean of {names[-1]} = {res.get('mean_b'):.4f}; n = {res.get('n_a')} and {res.get('n_b')}."))
    results.append(("p", _interpretation(verdict)))
    refs = [f"[{e.get('id')}] {e.get('title')}. {e.get('url', '')}".strip() for e in record.get("evidence") or []]
    return Draft("paper", f"Draft report: {record.get('topic')}", _banners(record, "paper"), abstract, [
        ("Introduction", intro), ("Method", method), ("Results", results),
        ("Limitations", _limitations(record)), ("References", [("bullets", refs or ["(none)"])])])


def _failed_run(record: Dict[str, Any]) -> Draft:
    execu, critic = record.get("execution") or {}, record.get("critic") or {}
    status = record.get("execution_status", "NOT_RUN")
    hyp, exp = record.get("hypothesis") or {}, record.get("experiment") or {}
    reason = ("No research paper was written, because there are no verified results to report. "
              f"Execution status: {status}; critic verdict: {critic.get('verdict', 'not run')}.")
    what: List[Block] = []
    if record.get("status") == "NO_EVIDENCE" or not hyp:
        what.append(("p", "The run stopped before a hypothesis was made, because no relevant papers were found in the "
                          "local index. Ingest papers on this topic first."))
    else:
        what += [("p", "Hypothesis (model-written): " + str(hyp.get("statement"))),
                 ("p", "Design: " + str(exp.get("design_summary") or "(not recorded)"))]
    happened: List[Block] = [("p", execu.get("message") or FAILED_ADVICE.get(status, "The run did not finish."))]
    attempts = execu.get("attempts") or []
    for a in attempts:
        res = a.get("result") or {}
        happened.append(("p", f"Attempt {a.get('number')}: {res.get('status')} (exit {res.get('exit_code')}, "
                              f"{res.get('duration_seconds')} s). {a.get('note') or ''}".strip()))
    if attempts:
        tail = str((attempts[-1].get("result") or {}).get("stderr", "")).strip()[-1200:]
        if tail:
            happened += [("p", "End of the last error output:"), ("code", tail)]
    if status in FAILED_ADVICE:
        happened.append(("p", FAILED_ADVICE[status]))
    for f in critic.get("findings") or []:
        if f.get("severity") == "error":
            happened.append(("p", f"Critic: {f.get('message')}"))
    sections: List[Tuple[str, List[Block]]] = [("What was planned", what), ("What happened", happened)]
    if record.get("evidence"):
        sections.append(("Papers retrieved", _evidence_blocks(record)))
    warns = _warning_bullets(record)
    if warns:
        sections.append(("Warnings recorded in the run", [("bullets", warns)]))
    return Draft("failed_run", f"Failed run report: {record.get('topic')}", _banners(record, "failed_run"),
                 reason, sections)


def build_draft(record: Dict[str, Any]) -> Draft:
    return _paper(record) if draft_kind(record) == "paper" else _failed_run(record)


# ------------------------------------------------------------------ rendering
def to_markdown(d: Draft) -> str:
    out = [f"# {d.title}", ""] + [f"> {b}" for b in d.banners] + ["", "## Abstract" if d.kind == "paper" else "## Summary",
                                                                      "", d.abstract, ""]
    for heading, blocks in d.sections:
        out += [f"## {heading}", ""]
        for kind, content in blocks:
            if kind == "bullets":
                out += [f"- {x}" for x in content] + [""]
            elif kind == "quote":
                out += [f"> {x}" for x in str(content).splitlines() or [""]] + [""]
            elif kind == "code":
                out += ["```", str(content), "```", ""]
            else:
                out += [str(content), ""]
    return "\n".join(out).rstrip() + "\n"


_TEX_MAP = {"\u2212": "-", "\u2013": "--", "\u2014": "---", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u03b5": "epsilon", "\u2192": "->", "\u00d7": "x", "\u2264": "<=", "\u2265": ">="}


def _ascii(text: str) -> str:
    text = "".join(_TEX_MAP.get(ch, ch) for ch in text)
    return unicodedata.normalize("NFKD", text).encode("ascii", "replace").decode("ascii")


def tex_escape(text: Any) -> str:
    text = _ascii(str(text))
    text = text.replace("\\", "\x00")
    for ch, rep in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                    ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        text = text.replace(ch, rep)
    return text.replace("\x00", r"\textbackslash{}")


def _tex_url_split(text: str) -> str:
    """Escape ordinary text but keep http(s) addresses inside \\url{}."""
    parts = re.split(r"(https?://\S+)", text)
    return "".join(r"\url{" + _ascii(p).replace("}", "%7D").replace("{", "%7B") + "}" if p.startswith("http") else tex_escape(p)
                   for p in parts)


def to_latex(d: Draft) -> str:
    out = [r"\documentclass{article}", r"\usepackage[T1]{fontenc}", r"\usepackage{url}",
           rf"\title{{{tex_escape(d.title)}}}", r"\author{AI-assisted draft (unverified)}", r"\date{}",
           r"\begin{document}", r"\maketitle", r"\begin{itemize}"]
    out += [rf"\item {tex_escape(b)}" for b in d.banners] + [r"\end{itemize}"]
    if d.kind == "paper":
        out += [r"\begin{abstract}", _tex_url_split(d.abstract), r"\end{abstract}"]
    else:
        out += [r"\section*{Summary}", _tex_url_split(d.abstract)]
    for heading, blocks in d.sections:
        out.append(rf"\section{{{tex_escape(heading)}}}")
        for kind, content in blocks:
            if kind == "bullets":
                out += [r"\begin{itemize}"] + [rf"\item {_tex_url_split(x)}" for x in content] + [r"\end{itemize}"]
            elif kind == "quote":
                out += [r"\begin{quote}", _tex_url_split(str(content)), r"\end{quote}"]
            elif kind == "code":
                out += [r"\begin{quote}\small\begin{verbatim}", _ascii(str(content)).replace(r"\end{verbatim}", ""),
                        r"\end{verbatim}\end{quote}"]
            else:
                out.append(_tex_url_split(str(content)) + "\n")
    out.append(r"\end{document}")
    return "\n".join(out) + "\n"


def write_draft(record_path: Path, out_dir: Path, fmt: str = "both") -> Tuple[str, List[Path]]:
    if fmt not in ("md", "tex", "both"):
        raise ValueError("format must be md, tex or both")
    record = load_record(Path(record_path))
    draft = build_draft(record)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(record_path).stem}_{'draft' if draft.kind == 'paper' else 'failed_run_report'}"
    files: List[Path] = []
    if fmt in ("md", "both"):
        files.append(out_dir / f"{stem}.md")
        files[-1].write_text(to_markdown(draft), encoding="utf-8")
    if fmt in ("tex", "both"):
        files.append(out_dir / f"{stem}.tex")
        files[-1].write_text(to_latex(draft), encoding="utf-8")
    return draft.kind, files
