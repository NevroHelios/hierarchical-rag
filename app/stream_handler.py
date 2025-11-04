from typing import Callable, Dict, Any
import streamlit as st
import time

class GraphStreamHandler:
    """Handler for streaming graph execution updates to Streamlit"""
    
    def __init__(self, flow_placeholder, status_placeholder, display_flow_func):
        self.flow_placeholder = flow_placeholder
        self.status_placeholder = status_placeholder
        self.display_flow_func = display_flow_func
        self.workers = []
        
    def on_node_start(self, node_name: str):
        """Called when a node starts execution"""
        pass
    
    def on_node_end(self, node_name: str, output: Dict[str, Any]):
        """Called when a node completes execution"""
        
        if node_name == "master_query":
            self.workers = list(output.get('queries', {}).keys())
            with self.flow_placeholder.container():
                self.display_flow_func(self.workers if self.workers else ['clinic', 'research', 'book'], 
                                      current_step="master_query")
            self.status_placeholder.info(f"🟦 **Master Query Node**: Generated {len(self.workers)} queries")
            time.sleep(0.3)
            
            # Show routing
            with self.flow_placeholder.container():
                self.display_flow_func(self.workers, current_step="routing")
            self.status_placeholder.info(f"🔀 **Routing**: Dispatching to → {', '.join(self.workers)}")
            time.sleep(0.3)
        
        elif node_name in ['clinic', 'research', 'book']:
            with self.flow_placeholder.container():
                self.display_flow_func(self.workers, current_step=node_name)
            
            icon = {'clinic': '🏥', 'research': '🔬', 'book': '📚'}.get(node_name, '📊')
            self.status_placeholder.info(f"{icon} **{node_name.capitalize()} Worker**: Retrieved context")
            time.sleep(0.3)
        
        elif node_name == "master_synthesizer":
            with self.flow_placeholder.container():
                self.display_flow_func(self.workers, current_step="synthesizer")
            self.status_placeholder.info("🟧 **Master Synthesizer**: Generating final answer...")
            time.sleep(0.3)
    
    def on_complete(self):
        """Called when graph execution completes"""
        with self.flow_placeholder.container():
            self.display_flow_func(self.workers, current_step="complete")
        self.status_placeholder.success("✅ **Complete**: Answer generated successfully!")
