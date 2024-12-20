import streamlit as st
import time
from typing import Optional

class LoadingAnimation:
    def __init__(self):
        if "dot_count" not in st.session_state:
            st.session_state.dot_count = 0
            
    def _get_dots(self) -> str:
        """Get the current dots animation frame."""
        dots = "." * (st.session_state.dot_count + 1)
        spaces = " " * (3 - len(dots))
        return dots + spaces

    def render(self, message: str = "Processing", key: Optional[str] = None):
        """Render the loading animation with message."""

        container_key = f"loading_container_{key}" if key else "loading_container"
        dots_key = f"dots_{key}" if key else "dots"
        
        if "last_update" not in st.session_state:
            st.session_state.last_update = time.time()
            
        current_time = time.time()
        if current_time - st.session_state.last_update >= 0.5:  # Update every 500ms
            st.session_state.dot_count = (st.session_state.dot_count + 1) % 3
            st.session_state.last_update = current_time

        loading_text = f"{message}{self._get_dots()}"

        return st.empty().markdown(f"```\n{loading_text}\n```")