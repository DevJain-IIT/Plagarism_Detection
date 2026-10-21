import streamlit as st

from plag.model import THRESHOLD, Detector

st.set_page_config(page_title="Semantic Plagiarism Detection", page_icon="🔍")


@st.cache_resource
def get_detector():
    # cached so the model is loaded only once
    return Detector()


st.title("Semantic Plagiarism Detection")
st.caption("Checks whether the candidate sentence restates the source. It was trained on "
           "an SNLI entailment proxy, so it compares sentence pairs, not whole documents.")

source = st.text_area("Source sentence",
                      "A man in a yellow jacket is standing on a rock in front of mountains.")
candidate = st.text_area("Candidate sentence", "A man in a yellow jacket is outdoors.")

if st.button("Check", type="primary") and source.strip() and candidate.strip():
    score = get_detector().score(source, candidate)
    if score >= THRESHOLD:
        st.error(f"Plagiarised (score {score:.2f})")
    else:
        st.success(f"Not plagiarised (score {score:.2f})")
    st.progress(score)
