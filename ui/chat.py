# Chat interface logic will go here.

import streamlit as st

def display_chat_history():
    """Display the conversation history"""
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong><br>{message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ai-message">
                <strong>Vaakeel Saab:</strong><br>{message["content"]}
            </div>
            """, unsafe_allow_html=True)