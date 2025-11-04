import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.graph import compile_graph
from app.states import MasterAgentState


st.set_page_config(page_title="Hierarchical RAG Demo", page_icon="📚")

st.title("📚 Hierarchical RAG — Multi-Domain Retrieval Demo")
st.markdown("Ask a question — the model will query across **Research**, **Clinical**, and **Books** domains!")

@st.cache_resource
def precompile_graph():
    app = compile_graph()
    return app
app = precompile_graph()

def ask_question(question: str) -> MasterAgentState:
    initial_state: MasterAgentState = {
        'question': question,
        'queries': {},
        'contexts': [],
        'answer': ''
    }
    final_state = app.invoke(initial_state)
    return final_state


question = st.text_area("Enter your question:")
debug = st.checkbox("Show reasoning trace", value=True)

if st.button("Submit") and question.strip():
    with st.spinner("Thinking..."):
        result = ask_question(question)

    st.success(result["answer"])

    if debug:
        with st.expander("Show Queries Made", expanded=True):
            st.markdown("### ❓ Queries Made")
            for agent, query in result["queries"].items():
              st.markdown(f"**{agent}:** {query}")
        with st.expander("Show Reasoning Trace", expanded=True):
            st.markdown("### 🧩 Reasoning Trace")
            for src, ctx in result["contexts"]:
              st.markdown(f"**{src}:** {ctx}")    