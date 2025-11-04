import sys
import os
import streamlit as st
import streamlit_mermaid as stmd
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.graph import compile_graph
from app.states import MasterAgentState
from app.stream_handler import GraphStreamHandler


st.set_page_config(page_title="Hierarchical RAG Demo", page_icon="📚", layout="wide")

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

def ask_question_streaming(question: str, flow_placeholder, status_placeholder):
    """Ask question and stream results step by step"""
    initial_state: MasterAgentState = {
        'question': question,
        'queries': {},
        'contexts': [],
        'answer': ''
    }
    
    # Create stream handler
    handler = GraphStreamHandler(flow_placeholder, status_placeholder, display_execution_flow)
    
    current_state = initial_state
    
    # Stream through the graph execution
    for event in app.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"[STREAMING] Node: {node_name}")
            current_state = {**current_state, **node_output}
            handler.on_node_end(node_name, node_output)
    
    handler.on_complete()
    return current_state


def display_graph_architecture():
    """Display the hierarchical graph architecture"""
    st.markdown("### 🏗️ Graph Architecture")
    st.markdown("""
    The system uses a **hierarchical agent architecture**:
    - **Master Query Node** (Blue): Analyzes question and routes to workers
    - **Worker Nodes** (Green): Parallel retrieval from Clinical, Research, Book domains
    - **Master Synthesizer** (Orange): Combines contexts into final answer
    """)
    
    # Try to get graph PNG from LangGraph
    mermaid_code = """
graph TD
    A([Query]) --> B{Master Query};
    
    B -->|DB 1| C[Book Service];
    B -->|DB 2| D[Clinic Service];
    B -->|DB 3| E[Research Service];
    
    C --> F{Master Synthesizer};
    D --> F;
    E --> F;
    
    F --> G([Answer]);
    
    style A fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff;
    style B fill:#2196F3,stroke:#1565C0,stroke-width:3px,color:#fff;
    style C fill:#FF9800,stroke:#E65100,stroke-width:2px,color:#fff;
    style D fill:#FF9800,stroke:#E65100,stroke-width:2px,color:#fff;
    style E fill:#FF9800,stroke:#E65100,stroke-width:2px,color:#fff;
    style F fill:#9C27B0,stroke:#6A1B9A,stroke-width:3px,color:#fff;
    style G fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff;
"""
    stmd.st_mermaid(mermaid_code, height=800)
    return
        

def display_execution_flow(workers, current_step=None):
    """Display interactive execution flow with highlighted active nodes"""
    st.markdown("### 🔄 Execution Flow")
    
    steps = [
        ("master_query", "🧠 Master Query Node", "#4A90E2"),
        ("routing", "🔀 Query Distribution", "#9B59B6"),
    ]
    
    # Add worker steps
    worker_colors = {
        'clinic': '#50C878',
        'research': '#50C878', 
        'book': '#50C878'
    }
    for worker in workers:
        icon = {'clinic': '🏥', 'research': '🔬', 'book': '📚'}.get(worker, '📊')
        steps.append((worker, f"{icon} {worker.capitalize()} Worker", worker_colors[worker]))
    
    steps.append(("synthesizer", "🎯 Master Synthesizer", "#E27D4A"))
    steps.append(("complete", "✅ Complete", "#27AE60"))
    
    # Create flow visualization
    cols = st.columns(len(steps))
    for idx, (step_id, step_name, color) in enumerate(steps):
        with cols[idx]:
            is_active = (current_step == step_id)
            opacity = "1.0" if is_active else "0.4"
            border = "3px solid #FFD700" if is_active else "2px solid #ddd"
            
            st.markdown(f"""
            <div style="
                background: {color};
                color: white;
                padding: 15px 10px;
                border-radius: 8px;
                text-align: center;
                opacity: {opacity};
                border: {border};
                font-weight: bold;
                font-size: 12px;
                min-height: 80px;
                display: flex;
                align-items: center;
                justify-content: center;
            ">
                {step_name}
            </div>
            """, unsafe_allow_html=True)
            
            if idx < len(steps) - 1:
                st.markdown("<div style='text-align: center; font-size: 20px;'>↓</div>", unsafe_allow_html=True)


# Add two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    display_graph_architecture()

with col2:
    question = st.text_area("Enter your question:", height=100)
    debug = st.checkbox("Show reasoning trace", value=True)
    
    if st.button("Submit", type="primary") and question.strip():
        # Create placeholders for real-time updates
        flow_container = st.container()
        result_container = st.container()
        
        with flow_container:
            flow_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Initialize display
            with flow_placeholder.container():
                display_execution_flow(['clinic', 'research', 'book'], current_step="master_query")
            status_placeholder.info("🚀 **Starting**: Initializing graph execution...")
            
            # Stream execution
            with st.spinner("Processing..."):
                result = ask_question_streaming(question, flow_placeholder, status_placeholder)
        
        with result_container:
            st.markdown("---")
            st.markdown("### 💡 Final Answer")
            st.success(result["answer"])

            if debug:
                st.markdown("---")
                with st.expander("📋 Show Queries Made", expanded=False):
                    for agent, query in result["queries"].items():
                        st.markdown(f"**{agent.capitalize()}:** `{query}`")
                
                with st.expander("📚 Show Retrieved Contexts", expanded=False):
                    for idx, (src, ctx) in enumerate(result["contexts"]):
                        st.markdown(f"#### {src.capitalize()} Context")
                        st.text_area(f"{src}_context", ctx, height=150, label_visibility="collapsed", key=f"context_{src}_{idx}")