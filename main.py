import os
import time
import uuid
import streamlit as st
from dotenv import load_dotenv

# --- Page Configuration ---
st.set_page_config(
    page_title="Vaakeel Saab",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Imports ---
from ui.chat import display_chat_history
from ui.faq import display_faq
from legal.vectorstore import setup_vectorstore, vectorize_data
from legal.gemini import gemini_chat
from legal.utils import generate_hash

# --- Load environment variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY not found. Please add it to your .env file.")
    st.stop()

import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)

# --- Model Configuration ---
DEFAULT_MODEL = "gemini-1.5-flash"  # Updated to Flash 2.0 equivalent
FALLBACK_MODEL = "gemini-1.0-pro"  # Fallback if quota exceeded

# --- Path Setup ---
working_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(working_dir, "data")
os.makedirs(data_dir, exist_ok=True)
vector_db_dir = os.path.join(working_dir, "vector_db_dir")
os.makedirs(vector_db_dir, exist_ok=True)

# --- Session State Initialization ---
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_vectorized" not in st.session_state:
    st.session_state.documents_vectorized = False
if "processing" not in st.session_state:
    st.session_state.processing = False
if "error" not in st.session_state:
    st.session_state.error = None
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0
if "current_model" not in st.session_state:
    st.session_state.current_model = DEFAULT_MODEL

# --- Rate Limiting ---
REQUEST_DELAY = 1.2  # seconds between requests

# --- Custom CSS ---
with open(os.path.join(working_dir, "assets", "style.css"), "r") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

# --- Navigation Bar ---
st.markdown("""
<nav style='position:sticky;top:0;z-index:100;display:flex;gap:28px;align-items:center;justify-content:center;background:rgba(30,30,30,0.56);backdrop-filter:blur(12px) saturate(120%);border-radius:18px;margin:20px 0 30px 0;padding:10px 38px 10px 38px;box-shadow:0 2px 18px 0 rgba(0,0,0,0.12);width:max-content;'>
    <a href="#upload-section" style='color:#fff;text-decoration:none;font-weight:600;font-size:1.08rem;padding:6px 18px;border-radius:8px;transition:background 0.2s;' onmouseover="this.style.background='rgba(29,233,182,0.13)'" onmouseout="this.style.background='none'">Upload Docs</a>
    <a href="#faq-section" style='color:#fff;text-decoration:none;font-weight:600;font-size:1.08rem;padding:6px 18px;border-radius:8px;transition:background 0.2s;' onmouseover="this.style.background='rgba(29,233,182,0.13)'" onmouseout="this.style.background='none'">Common FAQs</a>
    <a href="#about-section" style='color:#fff;text-decoration:none;font-weight:600;font-size:1.08rem;padding:6px 18px;border-radius:8px;transition:background 0.2s;' onmouseover="this.style.background='rgba(29,233,182,0.13)'" onmouseout="this.style.background='none'">About</a>
</nav>
""", unsafe_allow_html=True)

# --- Hero Section ---
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:18px;margin-bottom:18px;'>
        <span style='font-size:2.7rem;background: linear-gradient(90deg,#1de9b6 40%,#fff 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>⚖️</span>
        <span style='font-size:2.5rem;font-weight:800;letter-spacing:0.5px;background: linear-gradient(90deg,#fff 60%,#1de9b6 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>Vaakeel Saab</span>
    </div>
    <div style='font-size:1.18rem;font-weight:600;color:#e6e6e6;margin-bottom:10px;'>AI-Powered Legal Assistant for India</div>
    <div style='font-size:1.07rem;color:#b0b8c1;margin-bottom:10px;'>
        Get instant legal guidance powered by Gemini Flash 2.0 with 128K context window.
    </div>
    <ul style='font-size:1.07rem;color:#e0e0e0;margin-left:1.2em;margin-bottom:0.5em;'>
        <li>⚡ Faster than traditional legal research</li>
        <li>📚 Knowledge of Indian Constitution & Laws</li>
        <li>🔍 Document analysis capabilities</li>
        <li>🆓 Free basic legal guidance</li>
    </ul>
    <div style='margin-top:14px;font-size:1rem;color:#7ddcd3;background:rgba(29,233,182,0.07);padding:8px 18px;border-radius:10px;width:max-content;'>
        <strong>Current Model:</strong> {st.session_state.current_model}
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.image("assets/ai2.png", width=320)

# --- Document Upload Section ---
st.markdown("<div id='upload-section'></div>", unsafe_allow_html=True)
st.markdown("### 📄 Upload Legal Documents")
uploaded_files = st.file_uploader(
    "Supported formats: PDF, DOCX, TXT",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="Upload contracts, judgments, or legal notices (Max 50MB total)"
)

process_col1, process_col2 = st.columns(2)
with process_col1:
    if st.button("🚀 Process Documents", disabled=st.session_state.processing):
        if uploaded_files:
            total_size = sum([f.size for f in uploaded_files])
            if total_size > 50 * 1024 * 1024:  # 50MB limit
                st.error("Total size exceeds 50MB limit")
            else:
                st.session_state.processing = True
                try:
                    with st.spinner("Analyzing documents with AI..."):
                        success, message = vectorize_data(
                            data_dir, 
                            vector_db_dir, 
                            st.session_state.user_id, 
                            uploaded_files
                        )
                        if success:
                            st.session_state.documents_vectorized = True
                            st.toast("✅ Documents processed successfully!", icon="✅")
                        else:
                            st.error(f"❌ {message}")
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                finally:
                    st.session_state.processing = False
        else:
            st.warning("Please upload documents first")
with process_col2:
    if st.button("🧹 Clear Documents", disabled=st.session_state.processing):
        try:
            user_data_dir = os.path.join(data_dir, st.session_state.user_id)
            user_vector_dir = os.path.join(vector_db_dir, st.session_state.user_id)
            
            if os.path.exists(user_data_dir):
                for file in os.listdir(user_data_dir):
                    os.remove(os.path.join(user_data_dir, file))
            if os.path.exists(user_vector_dir):
                import shutil
                shutil.rmtree(user_vector_dir)
            
            st.session_state.documents_vectorized = False
            st.toast("🗑️ Documents cleared successfully!", icon="🗑️")
        except Exception as e:
            st.error(f"Error clearing documents: {str(e)}")

# --- Chat Interface ---
st.markdown("<div id='ask-section'></div>", unsafe_allow_html=True)
st.markdown("### 💬 Ask Vaakeel Saab")

chat_container = st.container()
with chat_container:
    display_chat_history()

user_query = st.chat_input("Type your legal question here...")
if user_query:
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_request_time
    
    # Rate limiting
    if time_since_last < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - time_since_last)
    
    st.session_state.last_request_time = time.time()
    
    # Add user message to chat history
    if {"role": "user", "content": user_query} not in st.session_state.chat_history:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with chat_container:
            display_chat_history()
        
        # Retrieve relevant document context if available
        context = None
        if st.session_state.documents_vectorized:
            try:
                vectorstore = setup_vectorstore(
                    user_specific=True,
                    vector_db_dir=vector_db_dir,
                    user_id=st.session_state.user_id
                )
                if vectorstore:
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                    docs = retriever.get_relevant_documents(user_query)
                    if docs:
                        context = "\n\n".join([
                            f"📄 Document Excerpt {i+1}:\n{doc.page_content[:2000]}..." 
                            for i, doc in enumerate(docs)
                        ])
            except Exception as e:
                st.error(f"Document retrieval error: {str(e)}")
        
        # Generate AI response with error handling
        try:
            with st.status(f"Analyzing with {st.session_state.current_model}...", expanded=True) as status:
                try:
                    ai_response = gemini_chat(
                        user_query,
                        context=context,
                        model_name=st.session_state.current_model
                    )
                    status.update(label="Analysis Complete", state="complete", expanded=False)
                except Exception as e:
                    if "quota" in str(e).lower():
                        st.session_state.current_model = FALLBACK_MODEL
                        st.warning(f"Switched to {FALLBACK_MODEL} due to API limits")
                        ai_response = gemini_chat(
                            user_query,
                            context=context,
                            model_name=FALLBACK_MODEL
                        )
                    else:
                        ai_response = f"⚠️ System Error: {str(e)}"
                    status.update(label="Completed with warnings", state="error")
        except Exception as e:
            ai_response = f"⚠️ Processing Error: {str(e)}"
        
        # Add AI response to chat history
        if {"role": "assistant", "content": ai_response} not in st.session_state.chat_history:
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            with chat_container:
                display_chat_history()

# --- FAQ Section ---
st.markdown("<div id='faq-section'></div>", unsafe_allow_html=True)
st.markdown("### 📚 Frequently Asked Questions")
display_faq()

# --- About Section ---
st.markdown("<div id='about-section'></div>", unsafe_allow_html=True)
st.markdown("""
<div class='glass-card' style='margin-top:24px;'>
    <span style='font-size:1.25rem;font-weight:700;color:#fff;'>About Vaakeel Saab</span>
    <div style='font-size:1.09rem;color:#e0e0e0;margin-top:10px;'>
        <p>Developed by <strong>Mallik Vinukonda</strong>, this application leverages cutting-edge AI to democratize access to legal information in India.</p>
        
        <p><strong>Key Features:</strong></p>
        <ul>
            <li>Powered by Google's Gemini Flash 2.0 AI</li>
            <li>Specialized in Indian legal framework</li>
            <li>Document analysis with 128K context window</li>
            <li>Cost-efficient operations</li>
        </ul>
        
        <p><strong>Disclaimer:</strong> This tool provides general legal information, not professional legal advice. For complex matters, please consult a qualified advocate.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='display:flex;justify-content:space-between;align-items:center;font-size:0.9rem;color:#b0b8c1;'>
    <div>© 2025 Vaakeel Saab | v2.1.0</div>
    <div>
        <span style='margin-right:15px;'>📧 contact@vaakeelsaab.in</span>
        <span>🔒 Your data remains private</span>
    </div>
</div>
""", unsafe_allow_html=True)
