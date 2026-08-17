import streamlit as st

from src.retrieval.hybrid_search import (
    load_resources,
    run_query,
)

from src.generation.answer_generator import (
    prepare_answer_request,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="3GPP RAG Assistant",
    page_icon="📡",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("📡 3GPP TS 38.331 RAG Assistant")

st.caption(
    "Grounded technical assistant using FAISS + BM25 + "
    "cross-encoder reranking + hallucination guard + "
    "local Qwen 2.5 1.5B."
)


# ============================================================
# LOAD RESOURCES ONCE
# ============================================================

@st.cache_resource
def load_rag_resources():

    return load_resources()


with st.spinner("Loading 3GPP RAG system..."):

    resources = load_rag_resources()


st.success("RAG system ready.")


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER QUESTION
# ============================================================

query = st.chat_input(
    "Ask a 3GPP TS 38.331 question..."
)


if query:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):

        st.markdown(query)


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching verified 3GPP evidence..."
        ):

            try:

                # ============================================
                # RETRIEVAL
                # ============================================

                results, validation_result = run_query(
                    query,
                    *resources,
                )


                # ============================================
                # ANSWER GENERATION
                # ============================================

                response = prepare_answer_request(
                    query,
                    results,
                    validation_result,
                )


                status = response.get(
                    "status",
                    "UNKNOWN"
                )

                answer = response.get(
                    "answer",
                    ""
                )


                # ============================================
                # STATUS
                # ============================================

                if status == "SUPPORTED":

                    st.success(
                        "✓ Answer supported by retrieved 3GPP evidence"
                    )

                elif status == "UNSUPPORTED":

                    st.warning(
                        "⚠ Insufficient verified 3GPP evidence"
                    )

                else:

                    st.error(
                        "LLM generation error"
                    )


                # ============================================
                # ANSWER
                # ============================================

                st.markdown(answer)


                # ============================================
                # CITATIONS
                # ============================================

                citations = response.get(
                    "citations",
                    []
                )

                if citations:

                    st.markdown(
                        "### 📚 3GPP Evidence"
                    )

                    for citation in citations:

                        section = citation.get(
                            "section",
                            ""
                        )

                        title = citation.get(
                            "title",
                            ""
                        )

                        page = citation.get(
                            "page",
                            ""
                        )

                        chunk_id = citation.get(
                            "chunk_id",
                            ""
                        )

                        st.markdown(
                            f"""
**Section:** {section}  
**Title:** {title}  
**Page:** {page}  
**Chunk:** `{chunk_id}`
"""
                        )


                # ============================================
                # RETRIEVAL DETAILS
                # ============================================

                with st.expander(
                    "🔎 Retrieval & Guard Details"
                ):

                    st.write(
                        "Validation status:",
                        validation_result.get(
                            "status"
                        ),
                    )

                    st.write(
                        "Reason:",
                        validation_result.get(
                            "reason"
                        ),
                    )

                    st.write(
                        "Requested identifiers:",
                        validation_result.get(
                            "identifiers",
                            [],
                        ),
                    )

                    st.write(
                        "Covered identifiers:",
                        validation_result.get(
                            "covered_identifiers",
                            [],
                        ),
                    )

                    st.write(
                        "Missing identifiers:",
                        validation_result.get(
                            "missing_identifiers",
                            [],
                        ),
                    )

                    st.write(
                        "Evidence records:",
                        validation_result.get(
                            "evidence_records",
                            0,
                        ),
                    )


            except Exception as exc:

                answer = (
                    "An error occurred while processing "
                    "the question."
                )

                st.error(
                    f"{type(exc).__name__}: {exc}"
                )


    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )