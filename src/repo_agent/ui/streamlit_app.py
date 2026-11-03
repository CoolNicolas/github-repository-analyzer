"""Streamlit interface for inspecting public GitHub repositories."""

import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("REPO_AGENT_API_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="GitHub AI Agent", page_icon="🐙", layout="wide")


@st.cache_data(ttl=5, show_spinner=False)
def load_provider_status() -> dict | None:
    try:
        response = httpx.get(f"{API_BASE_URL}/api/status", timeout=3)
        response.raise_for_status()
        return response.json()
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError):
        return None


st.title("GitHub Repository Analyzer")
st.caption("Public repositories are retrieved read-only. Repository code is never executed.")


def show_safe_diagnostics(diagnostics: dict | None) -> None:
    if not diagnostics:
        return
    with st.expander("RAG diagnostic metadata"):
        for label, key in (
            ("Pipeline stage", "pipeline_stage"),
            ("Provider status", "provider_status"),
            ("Finish reason", "finish_reason"),
            ("Model", "model"),
            ("Prompt tokens", "prompt_tokens"),
            ("Completion tokens", "completion_tokens"),
            ("Total tokens", "total_tokens"),
            ("Rejection reason", "rejection_reason"),
        ):
            st.caption(f"{label}: {diagnostics.get(key)}")
        st.caption(f"Returned citation IDs: {diagnostics.get('returned_citation_ids', [])}")
        st.caption(f"Allowed citation IDs: {diagnostics.get('allowed_citation_ids', [])}")
        st.caption(
            "Semantic grounding evaluated: "
            f"{diagnostics.get('semantic_grounding_evaluated', False)}"
        )


def render_agent_trace(trace: list[dict], diagnostics: dict | None = None) -> None:
    for entry in trace:
        st.markdown(
            f"**Step {entry.get('step')} · {entry.get('tool_name')} · {entry.get('status')}**"
        )
        safe_args = entry.get("arguments", {})
        if safe_args:
            st.caption("Arguments: " + str(safe_args))
        if entry.get("message"):
            st.caption(entry["message"])
        for reference in entry.get("references", []):
            line_range = ""
            if reference.get("start_line") is not None:
                line_range = f" lines {reference['start_line']}-{reference['end_line']}"
            st.caption(f"{reference['path']}{line_range} · commit {reference['commit_sha']}")
    if diagnostics:
        st.caption(
            "Model response diagnostics: "
            f"type={diagnostics.get('response_type')}, "
            f"finish={diagnostics.get('finish_reason')}, "
            f"tool_calls={diagnostics.get('tool_call_count')}, "
            f"parsing={diagnostics.get('parsing_status')}, "
            f"error={diagnostics.get('error_category')}"
        )
        st.caption(
            "Context budget: "
            f"estimated={diagnostics.get('estimated_context_tokens')}, "
            f"unique evidence chunks={diagnostics.get('unique_evidence_chunks')}, "
            f"duplicate evidence chunks={diagnostics.get('duplicate_evidence_chunks')}, "
            f"remaining after answer reserve={diagnostics.get('remaining_context_tokens')}, "
            f"termination={diagnostics.get('termination_reason')}, "
            f"token estimates by request/tool-result={diagnostics.get('context_token_history', [])}"
        )
        if diagnostics.get("http_status") is not None:
            st.caption(
                "Provider error metadata: "
                f"HTTP {diagnostics.get('http_status')}, "
                f"type={diagnostics.get('api_error_type')}, "
                f"code={diagnostics.get('api_error_code')}, "
                f"parameter={diagnostics.get('api_error_param')}, "
                f"request ID={diagnostics.get('request_id')}"
            )


provider_status = load_provider_status()
agent_enabled = bool(provider_status and provider_status.get("agent_enabled"))
if provider_status:
    st.caption(provider_status["message"])
else:
    st.caption("LLM provider status is unavailable. Start or reconnect to the FastAPI backend.")

with st.form("repository_form"):
    repository_url = st.text_input(
        "Public GitHub repository URL", placeholder="https://github.com/owner/repository"
    )
    submitted = st.form_submit_button("Inspect repository", type="primary")

if submitted:
    if not repository_url.strip():
        st.error("Enter a public GitHub repository URL.")
    else:
        try:
            response = httpx.post(
                f"{API_BASE_URL}/api/repos/inspect",
                json={"repository_url": repository_url.strip()},
                timeout=90,
            )
        except httpx.TimeoutException:
            st.error("The inspection request timed out. Try again or inspect a smaller repository.")
        except httpx.RequestError:
            st.error("Could not reach the FastAPI service. Start the API and try again.")
        else:
            if response.is_error:
                try:
                    error = response.json()
                    message = error.get("detail", "Repository inspection failed.")
                except ValueError:
                    message = "Repository inspection failed."
                st.error(message)
            else:
                result = response.json()
                st.session_state["inspection"] = result
                st.session_state["repository_url"] = repository_url.strip()
                st.session_state.pop("repository_analysis", None)
                st.session_state["question_history"] = []
                st.session_state["agent_history"] = []

inspection = st.session_state.get("inspection")
if inspection:
    repository = inspection["repository"]
    st.subheader(repository["full_name"])
    st.write(repository.get("description") or "No description provided.")
    st.markdown(f"[Open repository]({repository['html_url']})")
    overview_columns = st.columns(4)
    overview_columns[0].metric("Stars", f"{repository['stars']:,}")
    overview_columns[1].metric("Forks", f"{repository['forks']:,}")
    overview_columns[2].metric("Language", repository.get("language") or "Not specified")
    overview_columns[3].metric("Default branch", repository["default_branch"])
    st.caption(
        f"Default branch commit: `{repository['commit_sha']}` · "
        f"{repository['files_selected']} text files selected · "
        f"{repository['source_bytes']:,} source bytes"
    )
    if repository["tree_truncated"]:
        st.warning("GitHub truncated the file tree. The displayed file list may be incomplete.")

    st.subheader("Selected text files")
    if not inspection["files"]:
        st.info("No eligible text files were found within the configured size limits.")
    for file in inspection["files"]:
        with st.expander(f"{file['path']} · {file['size_bytes']:,} bytes"):
            st.caption(f"Blob `{file['blob_sha']}` · commit `{file['commit_sha']}`")
            st.markdown(f"[View this file at the inspected commit]({file['citation_url']})")
            st.code(file["content"], language="text", line_numbers=True)

    st.subheader("Repository analysis")
    with st.form("analysis_form"):
        analyze_submitted = st.form_submit_button("Analyze repository", type="primary")
    if analyze_submitted:
        with st.spinner("Analyzing the selected repository context…"):
            try:
                analysis_response = httpx.post(
                    f"{API_BASE_URL}/api/repos/analyze",
                    json={"repository_url": st.session_state["repository_url"]},
                    timeout=120,
                )
            except httpx.TimeoutException:
                st.error(
                    "The analysis request timed out. Try again or use a smaller context limit."
                )
            except httpx.RequestError:
                st.error("Could not reach the FastAPI service. Start the API and try again.")
            else:
                if analysis_response.is_error:
                    try:
                        error_payload = analysis_response.json()
                        message = error_payload.get("detail", "Repository analysis failed.")
                        show_safe_diagnostics(error_payload.get("diagnostics"))
                    except ValueError:
                        message = "Repository analysis failed."
                    st.error(message)
                else:
                    st.session_state["repository_analysis"] = analysis_response.json()

    if result := st.session_state.get("repository_analysis"):
        if not result["context_sufficient"]:
            st.warning("The available repository context is insufficient for a reliable analysis.")
        st.markdown(result["answer"])
        st.caption(f"Analysis used approximately {result['context_tokens']:,} context tokens.")

    st.subheader("Ask a question")
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Question about the selected repository",
            placeholder="How does the application handle configuration?",
        )
        question_submitted = st.form_submit_button("Ask")
    if question_submitted:
        if not question.strip():
            st.error("Enter a question about the repository.")
        else:
            with st.spinner("Searching the selected repository context…"):
                try:
                    answer_response = httpx.post(
                        f"{API_BASE_URL}/api/repos/ask",
                        json={
                            "repository_url": st.session_state["repository_url"],
                            "question": question.strip(),
                        },
                        timeout=120,
                    )
                except httpx.TimeoutException:
                    st.error("The question request timed out. Please try again.")
                except httpx.RequestError:
                    st.error("Could not reach the FastAPI service. Start the API and try again.")
                else:
                    if answer_response.is_error:
                        try:
                            error_payload = answer_response.json()
                            message = error_payload.get("detail", "Question answering failed.")
                            show_safe_diagnostics(error_payload.get("diagnostics"))
                        except ValueError:
                            message = "Question answering failed."
                        st.error(message)
                    else:
                        history = st.session_state.setdefault("question_history", [])
                        history.append((question.strip(), answer_response.json()))

    for previous_question, previous_answer in reversed(
        st.session_state.get("question_history", [])
    ):
        with st.container(border=True):
            st.markdown(f"**You asked:** {previous_question}")
            if not previous_answer["context_sufficient"]:
                st.warning(
                    "The available repository context is insufficient for a reliable answer."
                )
            st.markdown(previous_answer["answer"])
            st.caption(
                f"Commit `{previous_answer['commit_sha']}` · "
                f"approximately {previous_answer['context_tokens']:,} context tokens"
            )
            with st.expander("RAG retrieval trace"):
                indexed_files = previous_answer.get("indexed_files", [])
                st.caption(f"Indexed files: {len(indexed_files)}")
                if indexed_files:
                    st.code("\n".join(indexed_files), language="text")

                for title, field in (
                    ("Retrieved chunks", "retrieved_chunks"),
                    ("Chunks sent to the LLM", "prompt_chunks"),
                ):
                    st.markdown(f"**{title}**")
                    chunks = previous_answer.get(field, [])
                    if not chunks:
                        st.caption("None")
                    for chunk in chunks:
                        st.caption(
                            f"{chunk['chunk_id']} · {chunk['path']} · lines {chunk['start_line']}-"
                            f"{chunk['end_line']} · commit {chunk['commit_sha']} · "
                            f"blob {chunk['blob_sha']}"
                        )
                show_safe_diagnostics(previous_answer.get("diagnostics"))
            if previous_answer.get("citations"):
                with st.expander("Retrieved source references"):
                    for citation in previous_answer["citations"]:
                        line_range = ""
                        if citation.get("start_line") is not None:
                            line_range = f" lines {citation['start_line']}-{citation['end_line']}"
                        st.markdown(
                            f"- `{citation.get('chunk_id', 'source')}` "
                            f"[`{citation['path']}`{line_range}]"
                            f"({citation['citation_url']}) · commit "
                            f"`{citation['commit_sha']}`"
                        )

    st.subheader("Read-only agent")
    if not agent_enabled:
        st.info(
            "Agent mode is disabled. Enable AGENT_ENABLED in the FastAPI environment to opt in."
        )
    with st.form("agent_question_form", clear_on_submit=True):
        agent_question = st.text_input(
            "Question for the repository agent",
            placeholder="Trace how configuration is loaded and used.",
        )
        agent_submitted = st.form_submit_button("Run agent", disabled=not agent_enabled)
    if agent_submitted:
        if not agent_question.strip():
            st.error("Enter a question for the repository agent.")
        else:
            with st.spinner("The read-only agent is inspecting repository evidence…"):
                try:
                    agent_response = httpx.post(
                        f"{API_BASE_URL}/api/repos/agent",
                        json={
                            "repository_url": st.session_state["repository_url"],
                            "question": agent_question.strip(),
                        },
                        timeout=180,
                    )
                except httpx.TimeoutException:
                    st.error("The agent request timed out. Please try again.")
                except httpx.RequestError:
                    st.error("Could not reach the FastAPI service. Start the API and try again.")
                else:
                    try:
                        agent_result = agent_response.json()
                    except ValueError:
                        agent_result = {}
                    if agent_response.is_error:
                        st.error(agent_result.get("detail", "The agent request failed."))
                        if agent_result.get("trace"):
                            with st.expander("Agent tool trace"):
                                render_agent_trace(
                                    agent_result["trace"], agent_result.get("diagnostics")
                                )
                        elif agent_result.get("diagnostics"):
                            render_agent_trace([], agent_result["diagnostics"])
                    else:
                        history = st.session_state.setdefault("agent_history", [])
                        history.append((agent_question.strip(), agent_result))

    for agent_asked_question, agent_result in reversed(st.session_state.get("agent_history", [])):
        with st.container(border=True):
            st.markdown(f"**Agent question:** {agent_asked_question}")
            if not agent_result["context_sufficient"]:
                st.warning("The agent could not establish sufficient repository evidence.")
            st.markdown(agent_result["answer"])
            st.caption(
                f"Commit `{agent_result['commit_sha']}` · {agent_result['steps']} steps · "
                f"{agent_result['tool_calls']} tool calls"
            )
            with st.expander("Agent tool trace"):
                render_agent_trace(agent_result.get("trace", []), agent_result.get("diagnostics"))

st.divider()
