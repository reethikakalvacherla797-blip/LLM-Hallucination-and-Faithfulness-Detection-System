import streamlit as st
from detector import detect_hallucination

st.set_page_config(
    page_title="LLM Hallucination & Faithfulness Detector",
    page_icon="🔎",
    layout="centered"
)

st.title("LLM Hallucination & Faithfulness Detection")
st.write(
    "Check whether an AI-generated answer is supported by the context you provide. "
    "This is an NLI-based estimate, not a guarantee of factual truth."
)

with st.expander("Example"):
    st.markdown(
        "**Context:** The Sun is a star at the center of our solar system.\n\n"
        "**Answer:** The Sun is a planet.\n\n"
        "The detector should assess whether the answer is supported, contradicted, "
        "or not established by the context."
    )

context = st.text_area(
    "Reference context",
    placeholder="Paste trusted information here...",
    height=180
)
answer = st.text_area(
    "AI-generated answer",
    placeholder="Paste the answer you want to check...",
    height=120
)

if st.button("Check answer", type="primary", use_container_width=True):
    if not context.strip() or not answer.strip():
        st.warning("Please enter both the context and the answer.")
    else:
        try:
            with st.spinner("Checking the answer... The first run may download the model."):
                result = detect_hallucination(context.strip(), answer.strip())

            st.subheader("Result")
            if result["status"] == "SUPPORTED":
                st.success("Status: SUPPORTED by the context")
            elif result["status"] == "CONTRADICTED":
                st.error("Status: CONTRADICTED by the context")
            elif result["status"] == "INSUFFICIENT EVIDENCE":
                st.warning("Status: INSUFFICIENT EVIDENCE in the context")
            else:
                st.info(f"Status: {result['status']}")

            st.caption(f"Model's top label: {result['model_label']}")
            st.write("### NLI scores")
            st.progress(result["entailment_score"], text=f"Entailment: {result['entailment_score']:.1%}")
            st.progress(result["contradiction_score"], text=f"Contradiction: {result['contradiction_score']:.1%}")
            st.progress(result["neutral_score"], text=f"Neutral: {result['neutral_score']:.1%}")
            st.info(
                "Interpretation: the model compares the supplied context with the answer. "
                "It does not independently verify whether the context itself is true."
            )
        except Exception as exc:
            st.error("The model could not complete the check.")
            st.exception(exc)
