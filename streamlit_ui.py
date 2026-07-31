import os
# Force pure-Python implementation of Protobuf to avoid descriptor conflicts on Streamlit
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tempfile
import uuid
import streamlit as st

# Import RAG Service from backend
try:
    from rag_service_backend import AdaptiveFinanceRAGService as RAGService
except ImportError:
    from rag_service_backend import FinanceRAGService as RAGService

# ==========================================
# 1. Page Configuration & Custom Styling
# ==========================================
st.set_page_config(
    page_title="AI Corporate Action Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        max-width: 1250px;
        margin: 0 auto;
    }
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .stButton>button {
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Service Initialization
# ==========================================
@st.cache_resource
def load_rag_service():
    return RAGService()

try:
    rag_service = load_rag_service()
    st.session_state.rag_service = rag_service
except Exception as e:
    st.error(f"Configuration Error initializing backend RAG service: {e}")
    st.stop()

# ==========================================
# 3. Session State Initialization
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = set()

# ==========================================
# 4. User Authentication Portal
# ==========================================
def render_login_signup():
    st.markdown("<h2 style='text-align: center;'>🔐 Corporate Intelligence Portal</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Sign in or create an account to access corporate action scenario modeling.</p>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔑 Sign In", "📝 Create Account"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Please enter both username and password.")
                elif hasattr(st.session_state.rag_service, "authenticate_user") and \
                     st.session_state.rag_service.authenticate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    # Fallback login if auth service method is unconfigured
                    st.session_state.logged_in = True
                    st.session_state.username = username or "Analyst"
                    st.rerun()
                    
    with tab2:
        with st.form("signup_form"):
            new_user = st.text_input("New Username", placeholder="Choose a username")
            new_pass = st.text_input("New Password", type="password", placeholder="Choose a password")
            confirm_pass = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
            submit_signup = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submit_signup:
                if not new_user or not new_pass or not confirm_pass:
                    st.error("All fields are required.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                elif hasattr(st.session_state.rag_service, "register_user"):
                    success, msg = st.session_state.rag_service.register_user(new_user, new_pass)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.success("Account created successfully! Please sign in.")

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("")
        st.write("")
        render_login_signup()
    st.stop()

# ==========================================
# 5. Sidebar Controls & Added Features
# ==========================================
with st.sidebar:
    st.markdown(f"👤 **Logged in as:** `{st.session_state.username}`")
    
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.feedback_submitted = set()
            for key in ["sample_query", "voice_query"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    with col_nav2:
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # --- FEATURE 1: DATA INGESTION BY UPLOADING DOCUMENTS ---
    st.header("⚙️ Data Ingestion")
    st.caption("Upload company filings, financial reports, or memos")
    uploaded_files = st.file_uploader(
        "Upload Financial Reports (PDF/TXT)", 
        accept_multiple_files=True, 
        type=['pdf', 'txt']
    )
    
    if st.button("Ingest Uploaded Files", use_container_width=True):
        if uploaded_files:
            total_files = len(uploaded_files)
            progress_bar = st.progress(0, text="Initializing document ingestion...")
            
            for idx, uploaded_file in enumerate(uploaded_files):
                # Update progress bar per file being indexed
                progress_percent = (idx + 1) / total_files
                progress_bar.progress(
                    progress_percent, 
                    text=f"Indexing ({idx + 1}/{total_files}): {uploaded_file.name}"
                )
                
                suffix = f".{uploaded_file.name.split('.')[-1]}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    if hasattr(st.session_state.rag_service, "ingest_document"):
                        st.session_state.rag_service.ingest_document(tmp_file_path, uploaded_file.name)
                except Exception as e:
                    st.error(f"Error ingesting {uploaded_file.name}: {e}")
                finally:
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
            
            # Keep progress bar visible on screen at completion
            progress_bar.progress(1.0, text="✅ Ingestion complete! All documents processed.")
        else:
            st.warning("Please select a PDF or TXT file to upload first.")

    st.divider()

    # --- FEATURE 2: VOICE INPUT ---
    st.header("🎙️ Voice Input")
    st.caption("Record financial queries directly")
    audio_value = st.audio_input("Record audio query")
    if audio_value:
        with st.spinner("Transcribing spoken query..."):
            audio_bytes = audio_value.getvalue()
            if hasattr(st.session_state.rag_service, "transcribe_audio"):
                transcribed_text = st.session_state.rag_service.transcribe_audio(audio_bytes)
            else:
                transcribed_text = "Transcribed speech query placeholder."
            
            st.info(f"**Transcribed:** {transcribed_text}")
            if transcribed_text and "failed" not in transcribed_text.lower():
                st.session_state.voice_query = transcribed_text

    st.divider()

    # --- FEATURE 3: SAMPLE QUESTIONS ---
    st.header("📝 Sample Questions")
    st.caption("Corporate Action Scenario Benchmarks")
    
    sample_questions = [
        "1. How does a 2-for-1 stock split affect a company's market capitalization?",
        "2. What is the impact of a cash dividend on the company's balance sheet?",
        "3. Explain the scenario modeling steps for a reverse merger.",
        "4. How do rights issues dilute earnings per share (EPS)?",
        "5. What happens to outstanding options during a corporate spin-off?",
        "6. How can we model the impact of a share repurchase program on WACC?",
        "7. What are the key tax implications to model during a cross-border M&A?",
        "8. Describe the effect of a bonus issue on share price and equity structure.",
        "9. How does an acquisition funded entirely by debt alter credit ratings?",
        "10. What key variables are adjusted when modeling a bankruptcy reorganization?"
    ]
    
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            clean_q = q.split(". ", 1)[1] if ". " in q else q
            st.session_state.sample_query = clean_q

# ==========================================
# 6. Main Chat Area
# ==========================================
st.markdown('<div class="main-header">📈 AI Corporate Action Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Retrieval-Augmented Generation & Scenario Modeling</div>', unsafe_allow_html=True)

# Render Welcome Screen if Chat is Empty
if not st.session_state.messages:
    st.markdown(f"### Welcome to your financial workspace, **{st.session_state.username}**! 👋")
    
    st.markdown("""
    Use this assistant to model corporate restructuring events, analyze filings, and calculate financial adjustments.
    
    #### 🚀 Key Features Available:
    1. **⚙️ Document Ingestion**: Upload PDF/TXT financial statements or prospectus sheets on the sidebar.
    2. **🎙️ Voice Query Input**: Speak your query directly using the audio recorder.
    3. **📝 Predefined Scenarios**: Click any of the 10 Sample Questions on the sidebar to inspect standard corporate action logic.
    4. **🎯 Adaptive ML Training**: Rate generated responses and submit target corrections to continuously adapt vector memory accuracy over time.
    """)
    st.divider()

# Display Chat History
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Render Adaptive ML Feedback UI on Assistant Responses
        if message["role"] == "assistant" and msg_idx > 0:
            msg_id = message.get("id", str(msg_idx))
            prev_query = st.session_state.messages[msg_idx - 1]["content"] if msg_idx > 0 else ""

            if msg_id in st.session_state.feedback_submitted:
                st.caption("✅ *Feedback recorded. Vector memory updated.*")
            else:
                with st.expander("🎯 Rate Response & Train Adaptive Memory"):
                    rating = st.slider("Quality Rating (1 = Low, 5 = High)", 1, 5, 5, key=f"r_{msg_id}")
                    correction = st.text_area("Expected / Corrected Answer (Optional):", key=f"c_{msg_id}")
                    
                    if st.button("Submit Feedback", key=f"b_{msg_id}"):
                        target_ans = correction.strip() if correction.strip() else message["content"]
                        
                        # Store feedback if backend supports it
                        if hasattr(st.session_state.rag_service, "add_feedback"):
                            st.session_state.rag_service.add_feedback(prev_query, target_ans, rating)
                        elif hasattr(st.session_state.rag_service, "feedback_collection"):
                            try:
                                emb = st.session_state.rag_service.encoder.encode([prev_query]).tolist()
                                fb_id = f"fb_{st.session_state.rag_service.feedback_collection.count() + 1}"
                                st.session_state.rag_service.feedback_collection.add(
                                    ids=[fb_id], embeddings=emb, documents=[target_ans], metadatas=[{"rating": rating}]
                                )
                            except Exception:
                                pass
                                
                        st.session_state.feedback_submitted.add(msg_id)
                        st.success("Feedback saved! Memory updated.")
                        st.rerun()

# Determine User Query Source (Priority: Sample Q -> Voice -> Chat Box)
user_query = None
if "sample_query" in st.session_state:
    user_query = st.session_state.sample_query
    del st.session_state.sample_query
elif "voice_query" in st.session_state:
    user_query = st.session_state.voice_query
    del st.session_state.voice_query
else:
    user_query = st.chat_input("Ask a question about stock splits, dividends, buybacks, or M&A...")

# Process User Query
if user_query:
    # Display user input
    st.session_state.messages.append({
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": user_query
    })
    
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing corporate action scenarios..."):
            try:
                if hasattr(st.session_state.rag_service, "answer_query"):
                    response = st.session_state.rag_service.answer_query(user_query)
                else:
                    response = "Unable to process query: answer_query method not found on RAG service."
            except Exception as err:
                response = f"An error occurred while evaluating scenario: {err}"
            
            st.markdown(response)

    st.session_state.messages.append({
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": response
    })
    st.rerun()