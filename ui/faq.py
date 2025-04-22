# FAQ section logic will go here.

import streamlit as st
from legal.gemini import gemini_chat

def display_faq():
    faq_data = [
        {"question": "How to file an FIR in India?", "category": "Criminal Procedure"},
        {"question": "What are my fundamental rights as an Indian citizen?", "category": "Constitutional Law"},
        {"question": "What is the process for filing a consumer complaint?", "category": "Consumer Law"},
        {"question": "How to respond to a legal notice?", "category": "Civil Procedure"},
        {"question": "What are tenant rights in India?", "category": "Property Law"},
        {"question": "How to apply for legal aid in India?", "category": "Legal Services"}
    ]
    faq_cols = st.columns(3)
    for i, faq in enumerate(faq_data):
        with faq_cols[i % 3]:
            st.markdown(f"""
            <div class=\"info-card\" style=\"height: 150px; cursor: pointer;\">
                <strong>{faq['question']}</strong><br>
                <span style=\"color: var(--accent);\">{faq['category']}</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Answer", key=f"faq_btn_{i}", help=f"Get answer to: {faq['question']}"):
                faq_answer = gemini_chat(faq["question"])
                st.markdown(f"""
                <div class=\"ai-message\">
                    <strong>Question:</strong> {faq['question']}<br><br>
                    <strong>Answer:</strong><br>{faq_answer}
                </div>
                """, unsafe_allow_html=True)
