import os
import time
import uuid
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(
    page_title="Vaakeel Saab",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# --- Ensure required directories exist ---
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

# --- Custom JS for Navbar Clicks ---
st.markdown('''
<script>
function setSection(section) {
    window.parent.postMessage({isStreamlitMessage: true, section: section}, '*');
}
document.querySelectorAll('nav a').forEach(function(link) {
    link.addEventListener('click', function(e) {
        e.preventDefault();
        const section = this.getAttribute('href').replace('#','');
        window.parent.postMessage({isStreamlitMessage: true, section: section}, '*');
    });
});
window.addEventListener('message', function(event) {
    if (event.data && event.data.isStreamlitMessage && event.data.section) {
        window.location.hash = event.data.section;
        window.scrollTo({top: 0, behavior: 'smooth'});
    }
});
</script>
''', unsafe_allow_html=True)

# --- Custom UI Styling ---
with open(os.path.join(working_dir, "assets", "style.css"), "r") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

# --- Glassy Navbar ---
st.markdown("""
<nav style='position:sticky;top:0;z-index:100;display:flex;gap:28px;align-items:center;justify-content:center;background:rgba(30,30,30,0.56);backdrop-filter:blur(12px) saturate(120%);border-radius:18px;margin:20px 0 30px 0;padding:10px 38px 10px 38px;box-shadow:0 2px 18px 0 rgba(0,0,0,0.12);width:max-content;'>
    <a href="#upload-section" style='color:#fff;text-decoration:none;font-weight:600;font-size:1.08rem;padding:6px 18px;border-radius:8px;transition:background 0.2s;' onmouseover="this.style.background='rgba(29,233,182,0.13)'" onmouseout="this.style.background='none'">Upload Docs</a>
    <a href="#faq-section" style='color:#fff;text-decoration:none;font-weight:600;font-size:1.08rem;padding:6px 18px;border-radius:8px;transition:background 0.2s;' onmouseover="this.style.background='rgba(29,233,182,0.13)'" onmouseout="this.style.background='none'">Common FAQs</a>
    <a href="#about-section" style='color:#fff;text-decoration:none;font-weight:600;font-size:1.08rem;padding:6px 18px;border-radius:8px;transition:background 0.2s;' onmouseover="this.style.background='rgba(29,233,182,0.13)'" onmouseout="this.style.background='none'">About</a>
</nav>
""", unsafe_allow_html=True)

# --- Hero Section Anchor ---
st.markdown("<div id='hero-section'></div>", unsafe_allow_html=True)

# --- HERO SECTION ---
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:18px;margin-bottom:18px;'>
        <span style='font-size:2.7rem;background: linear-gradient(90deg,#1de9b6 40%,#fff 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>⚖️</span>
        <span style='font-size:2.5rem;font-weight:800;letter-spacing:0.5px;background: linear-gradient(90deg,#fff 60%,#1de9b6 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>Your Indian Legal Assistant</span>
    </div>
    <div style='font-size:1.18rem;font-weight:600;color:#e6e6e6;margin-bottom:10px;'>AI-Powered Legal Help for Everyone</div>
    <div style='font-size:1.07rem;color:#b0b8c1;margin-bottom:10px;'>
        Upload legal documents or ask questions for instant, reliable insights based on Indian law.
    </div>
    <ul style='font-size:1.07rem;color:#e0e0e0;margin-left:1.2em;margin-bottom:0.5em;'>
        <li>Document analysis & summarization</li>
        <li>Legal Q&A from your uploads</li>
        <li>Constitutional law info</li>
        <li>Guidance on procedures & rights</li>
    </ul>
    <div style='margin-top:14px;font-size:1rem;color:#7ddcd3;background:rgba(29,233,182,0.07);padding:8px 18px;border-radius:10px;width:max-content;'>
        <strong>Note:</strong> Trained on Indian Constitution & public legal resources. For complex cases, consult a qualified advocate.
    </div>
    """, unsafe_allow_html=True)
with col2:
    # Use st.image for reliable local file display, with negative margin for alignment
    st.markdown("<div style='margin-top:-100px'></div>", unsafe_allow_html=True)
    st.image("assets/ai2.png", width=320)

# --- Upload Documents Section ---
st.markdown("<div id='upload-section'></div>", unsafe_allow_html=True)
st.markdown("### 📄 Upload Legal Documents for Instant Analysis")
uploaded_files = st.file_uploader(
    "Upload legal documents (PDF only)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload contracts, judgments, or any legal document for instant analysis and Q&A"
)
process_col1, process_col2 = st.columns(2)
with process_col1:
    if st.button("Process Documents", disabled=st.session_state.processing):
        if uploaded_files:
            success, message = vectorize_data(data_dir, vector_db_dir, st.session_state.user_id, uploaded_files)
            if success:
                st.success(message)
            else:
                st.error(message)
        else:
            st.warning("Please upload documents first.")
with process_col2:
    if st.button("Clear Documents", disabled=st.session_state.processing):
        user_data_dir = os.path.join(data_dir, st.session_state.user_id)
        user_vector_dir = os.path.join(vector_db_dir, st.session_state.user_id)
        try:
            if os.path.exists(user_data_dir):
                for file in os.listdir(user_data_dir):
                    os.remove(os.path.join(user_data_dir, file))
            if os.path.exists(user_vector_dir):
                import shutil
                shutil.rmtree(user_vector_dir)
            st.session_state.documents_vectorized = False
            st.success("Documents cleared successfully!")
        except Exception as e:
            st.error(f"Error clearing documents: {str(e)}")
if st.session_state.processing:
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px;'>
        <div class='loading-spinner'></div>
        <span>Processing documents...</span>
    </div>
    """, unsafe_allow_html=True)

# --- Chat interface ---
st.markdown("<div id='ask-section'></div>", unsafe_allow_html=True)
st.markdown("### 💬 Ask Vaakeel Saab a Question")
chat_container = st.container()
with chat_container:
    display_chat_history()
user_query = st.chat_input("Type your legal question here...")
if user_query:
    if {"role": "user", "content": user_query} not in st.session_state.chat_history:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with chat_container:
            display_chat_history()
        context = None
        if st.session_state.documents_vectorized:
            try:
                vectorstore = setup_vectorstore(user_specific=True, vector_db_dir=vector_db_dir, user_id=st.session_state.user_id)
                if vectorstore:
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                    docs = retriever.get_relevant_documents(user_query)
                    if docs:
                        context = "\n\n".join([f"Document excerpt {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
            except Exception as e:
                st.error(f"Error retrieving documents: {str(e)}")
        with st.status("Generating response..."):
            ai_response = gemini_chat(user_query, context)
        if {"role": "assistant", "content": ai_response} not in st.session_state.chat_history:
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            with chat_container:
                display_chat_history()

# --- FAQ Section ---
# Move anchor right before FAQ heading
st.markdown("<div id='faq-section'></div>", unsafe_allow_html=True)
st.markdown("### 📚 Common Legal Questions")
display_faq()

# --- About Section ---
st.markdown("<div id='about-section'></div>", unsafe_allow_html=True)
st.markdown("""
<div class='glass-card' style='margin-top:24px;'>
    <span style='font-size:1.25rem;font-weight:700;color:#fff;'>About the Developer</span>
    <div style='font-size:1.09rem;color:#e0e0e0;margin-top:10px;'>
        <p>Hi, I'm <strong>Mallik Vinukonda</strong>, a passionate Computer Science student with a deep interest in technology and its real-world impact. Growing up with a father who is a dedicated lawyer, I witnessed firsthand the challenges and opportunities in the legal field. Inspired by his work and driven by my love for coding, I created this AI-powered legal assistant to make legal information more accessible to everyone in India.</p>
        <p>This project blends my technical skills and personal experiences to help bridge the gap between law and technology for the benefit of all.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Footer / Disclaimer ---
st.markdown("---")
st.markdown("""
<div style='display: flex; justify-content: space-between; align-items: center; font-size: 1rem;'>
    <span> 2025 Vaakeel Saab | Version 1.0.0</span>
    <span><strong>Disclaimer:</strong> This tool is for informational purposes only. It does not constitute legal advice. For case-specific guidance, consult a qualified advocate.</span>
</div>
""", unsafe_allow_html=True)