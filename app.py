"""ThirdUmpire -- Streamlit UI (PROJECT_PLAN.md Phase 7).

No retrieval or prompt logic belongs here -- this only calls run_query()
and renders the AskResponse it returns.

Usage: streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.citations import find_matching_chunk
from src.config import BASE_DIR, MAX_QUESTION_CHARS, RERANKER_FT_PATH
from src.pipeline import run_query
from src.schemas import AskResponse

st.set_page_config(page_title="ThirdUmpire", page_icon=":material/sports_cricket:", layout="centered")

# (competition code, display label) -- None is the "let the query analyzer
# auto-detect it" option exposed in the CLI as no --competition flag.
_COMPETITION_OPTIONS: list[tuple[str | None, str]] = [
    (None, "Auto-detect"),
    ("laws", "Laws of Cricket"),
    ("ipl", "IPL"),
    ("icc_t20i", "ICC T20I"),
    ("icc_test", "ICC Test"),
    ("icc_odi", "ICC ODI"),
    ("bcci_domestic_multiday", "BCCI domestic multi-day"),
    ("bcci_domestic_odi", "BCCI domestic one-day"),
    ("bcci_domestic_t20", "BCCI domestic T20"),
]
_COMPETITION_LABELS = dict(_COMPETITION_OPTIONS)
_COMPETITION_CODES = {label: code for code, label in _COMPETITION_OPTIONS}

_APPLIES_TO_COLORS = {
    "laws": "gray",
    "ipl": "orange",
    "icc_t20i": "blue",
    "icc_test": "blue",
    "icc_odi": "blue",
    "bcci_domestic_multiday": "green",
    "bcci_domestic_odi": "green",
    "bcci_domestic_t20": "green",
    "unknown": "gray",
}


def _ft_reranker_available() -> bool:
    return (BASE_DIR / RERANKER_FT_PATH).exists()


with st.sidebar:
    st.subheader("Settings")

    competition_label = st.selectbox(
        "Competition",
        options=[label for _, label in _COMPETITION_OPTIONS],
        help="Auto-detect lets ThirdUmpire infer the competition from the question's wording.",
    )
    competition = _COMPETITION_CODES[competition_label]

    first_stage = st.segmented_control(
        "First stage",
        options=["dense", "hybrid"],
        default="dense",
        help="How candidate clauses are retrieved before reranking.",
    )

    reranker_options = ["base"] + (["fine-tuned"] if _ft_reranker_available() else [])
    reranker_label = st.segmented_control(
        "Reranker",
        options=reranker_options,
        default="base",
        help=(
            "base = cross-encoder/ms-marco-MiniLM-L-6-v2. fine-tuned = the Phase 6 model trained on this corpus."
            if _ft_reranker_available()
            else "Only the base reranker is available -- no fine-tuned model found at models/reranker-ft/."
        ),
    )
    reranker = "ft" if reranker_label == "fine-tuned" else "base"

    debug = st.toggle("Show debug info", help="Show retrieved vs reranked candidates, scores, and per-stage latencies.")

    st.caption("Rule documents belong to MCC, ICC and BCCI and aren't redistributed by this project.")

st.title("ThirdUmpire", icon=":material/sports_cricket:")
st.caption("Ask a cricket rules question. Answers are grounded with citations to the exact document, clause and page.")

with st.form("ask_form", border=False):
    with st.container(horizontal=True, vertical_alignment="bottom"):
        question = st.text_input(
            "Question",
            placeholder='e.g. "In the Ranji Trophy, by how many runs must the side batting first lead to enforce the follow-on?"',
            label_visibility="collapsed",
            max_chars=MAX_QUESTION_CHARS,
        )
        submitted = st.form_submit_button("Ask", icon=":material/send:", type="primary")

if submitted and question.strip():
    with st.spinner("Retrieving, reranking and asking Gemini..."):
        st.session_state["response"] = run_query(
            question.strip(),
            competition=competition,
            first_stage=first_stage or "dense",
            reranker=reranker,
            debug=debug,
        )
    st.session_state["question"] = question.strip()
elif submitted:
    st.warning("Type a question first.", icon=":material/error:")

response: AskResponse | None = st.session_state.get("response")

if response is not None:
    st.divider()
    st.caption(f"Q: {st.session_state['question']}")

    with st.container(horizontal=True):
        st.badge("Found" if response.answer.found else "Not found", color="green" if response.answer.found else "red")
        applies_to_display = _COMPETITION_LABELS.get(response.answer.applies_to, response.answer.applies_to)
        st.badge(applies_to_display, color=_APPLIES_TO_COLORS.get(response.answer.applies_to, "gray"))
        if not response.citation_valid:
            st.badge("Citation not fully validated", color="red", icon=":material/warning:")

    st.markdown(response.answer.answer)

    if not response.citation_valid:
        st.warning(
            "One or more citations could not be validated against the retrieved context. "
            "The answer may still be correct, but double-check the cited clause yourself.",
            icon=":material/warning:",
        )

    if response.answer.citations:
        st.subheader("Citations")
        for citation in response.answer.citations:
            chunk = find_matching_chunk(citation, response.context_chunks)
            is_invalid = citation in response.invalid_citations
            title = f"{citation.doc_id} | {citation.clause} | p.{citation.page}"
            if is_invalid:
                title += " -- could not be validated"
            with st.expander(title, icon=":material/error:" if is_invalid else ":material/description:"):
                if chunk is not None:
                    st.text(chunk.text)
                else:
                    st.caption("This citation doesn't match any chunk that was actually passed to the model.")

    if debug:
        st.subheader("Debug")

        with st.container(horizontal=True):
            for stage, seconds in response.latencies.items():
                st.metric(stage.capitalize(), f"{seconds:.2f}s")

        retrieved_tab, reranked_tab = st.tabs(["Retrieved (first stage)", "Reranked"])
        with retrieved_tab:
            for sc in response.retrieved or []:
                st.text(f"{sc.score:.3f}  {sc.chunk.doc_id}  {sc.chunk.clause}  {sc.chunk.clause_title}")
        with reranked_tab:
            for sc in response.reranked or []:
                raw = f"{sc.raw_score:.3f}" if sc.raw_score is not None else "n/a"
                st.text(f"{sc.score:.3f} (raw {raw})  {sc.chunk.doc_id}  {sc.chunk.clause}  {sc.chunk.clause_title}")
