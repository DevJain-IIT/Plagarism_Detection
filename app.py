import streamlit as st

from plag.model import THRESHOLD, Detector

st.set_page_config(page_title="Semantic Plagiarism Detection", page_icon="🔍")


@st.cache_resource
def detector():
    return Detector()


st.title("Semantic Plagiarism Detection")
st.caption("Does the candidate sentence restate the source? Trained on an SNLI entailment proxy, "
           "so it judges sentence pairs, not whole documents.")

source = st.text_area("Source sentence", "A man in a yellow jacket is standing on a rock in front of mountains.")
candidate = st.text_area("Candidate sentence", "A man in a yellow jacket is outdoors.")

if st.button("Check", type="primary") and source.strip() and candidate.strip():
    score = detector().score(source, candidate)
    (st.error if score >= THRESHOLD else st.success)(
        f"{'Plagiarised' if score >= THRESHOLD else 'Not plagiarised'} (similarity score {score:.2f})")
    st.progress(score)
