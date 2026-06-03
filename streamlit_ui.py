import os
# CRITICAL: Force the pure Python Protobuf implementation to bypass binary descriptor conflicts on Streamlit Cloud
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tempfile
import streamlit as st
from rag_service_backend import FinanceRAGService

# Page Configuration
st.set_page_config(page_title="AI-Driven Corporate Action Assistant-ChatBot", page_icon="📈", layout="wide")

# --- INITIALIZE RAG SERVICE ---
# Initialize the backend service FIRST so the login portal can access server auth APIs
if "rag_service" not in st.session_state:
    try:
        st.session_state.rag_service = FinanceRAGService()
    except ValueError as e:
        st.error(f"Configuration Error: {e}")
        st.stop()

# --- USER AUTHENTICATION SYSTEM ---
# Initialize Session States for Auth
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

def render_login_signup():
    """Renders the login/signup clean tab interface calling backend server APIs."""
    st.markdown("<h2 style='text-align: center;'>🔐 Welcome - login to continue</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>AI-Driven Corporate Action Assistant-ChatBot</p>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔑 Sign In", "📝 Create Account"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Please fill in all fields.")
                elif st.session_state.rag_service.authenticate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
                    
    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("New Username", placeholder="Choose a unique username")
            new_password = st.text_input("New Password", type="password", placeholder="Choose a strong password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
            submit_signup = st.form_submit_button("Sign Up & Create Account", use_container_width=True)
            
            if submit_signup:
                if not new_username or not new_password or not confirm_password:
                    st.error("All fields are required.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    # Offload registration logic entirely to the backend
                    success, msg = st.session_state.rag_service.register_user(new_username, new_password)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

# --- MAIN APP ROUTING ---
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("")
        st.write("")
        render_login_signup()
    st.stop()  # Stop rendering the app further if user is not authenticated

# --- MAIN WORKSPACE ---
# Initialize Chat History in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar configurations
with st.sidebar:
    # User Profile Section
    st.markdown(f"👤 **Logged in as:** `{st.session_state.username}`")
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.messages = []
            if "sample_query" in st.session_state:
                del st.session_state.sample_query
            if "voice_query" in st.session_state:
                del st.session_state.voice_query
            st.rerun()
    with col_nav2:
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.header("⚙️ Data Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Financial Reports (PDF/TXT)", 
        accept_multiple_files=True, 
        type=['pdf', 'txt']
    )
    
    if st.button("Ingest Files"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process via RAG service
                    try:
                        msg = st.session_state.rag_service.ingest_document(tmp_file_path, uploaded_file.name)
                        st.success(msg)
                    except Exception as e:
                        st.error(f"Error ingesting {uploaded_file.name}: {e}")
                    finally:
                        os.unlink(tmp_file_path) # Cleanup temp file
        else:
            st.warning("Please upload a file first.")

    st.divider()
    st.header("🎙️ Voice Input")
    audio_value = st.audio_input("Record a financial query")
    if audio_value:
        with st.spinner("Transcribing audio..."):
            audio_bytes = audio_value.getvalue()
            transcribed_text = st.session_state.rag_service.transcribe_audio(audio_bytes)
            st.info(f"**Transcribed:** {transcribed_text}")
            
            if transcribed_text and "failed" not in transcribed_text.lower():
                 st.session_state.voice_query = transcribed_text

    st.divider()
    st.header("📝 Sample Questions")
    st.caption("Corporate Action Scenario Modelling")
    
    sample_questions = [
        "1. How does a 2-for-1 stock split affect a company's market capitalization?",
        "2. What is the impact of a cash dividend on the company's balance sheet?",
        "3. Explain the scenario modeling steps for a reverse merger.",
        "4. How do rights issues dilute the earnings per share (EPS)?",
        "5. What happens to outstanding options during a spin-off?",
        "6. How can we model the impact of a share repurchase program on the WACC?",
        "7. What are the tax implications to model during a cross-border M&A?",
        "8. Describe the effect of a bonus issue on the share price and equity structure.",
        "9. How does an acquisition funded entirely by debt alter the target's credit rating scenario?",
        "10. What are the key variables to adjust when modeling a bankruptcy reorganization?"
    ]
    
    for q in sample_questions:
        if st.button(q, use_container_width=True):
             clean_q = q.split(". ", 1)[1] 
             st.session_state.sample_query = clean_q

# Main UI Chat Interface
st.title("📈 AI-Driven Corporate Action Assistant-ChatBot")

# RENDER WELCOME SCREEN (If no messages exist in the chat history)
if not st.session_state.messages:
    st.markdown(f"### Welcome to your corporate workspace, **{st.session_state.username}**! 👋")
    st.markdown("""
    This assistant is powered by Langchain, ChromaDB, and state-of-the-art open-access Language Models. 
    It is specifically engineered to model complex corporate action scenarios.
    
    #### 🚀 How to Get Started:
    1. **Upload Documents**: Use the **Data Ingestion** section on the sidebar to load company filings, memos, or financial prospectus sheets (PDF/TXT format).
    2. **Submit Queries**: You can write a query in the chat input below, or use the **🎙️ Voice Input** feature to dictate your question.
    3. **Run Predefined Scenarios**: Click on any of the **Sample Questions** on the sidebar to instantly observe how splits, dividends, rights issues, and mergers alter valuation/capital structure frameworks.
    
    Select a sample question on the left or type in the input bar below to begin!
    """)
    st.divider()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle inputs (Prioritize: Sample Q -> Voice -> Text Box)
user_query = None
if "sample_query" in st.session_state:
    user_query = st.session_state.sample_query
    del st.session_state.sample_query
elif "voice_query" in st.session_state:
    user_query = st.session_state.voice_query
    del st.session_state.voice_query
else:
    user_query = st.chat_input("Ask a question about corporate actions...")

# Process the query
if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing corporate scenarios..."):
            try:
                response = st.session_state.rag_service.answer_query(user_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Failed to generate answer: {str(e)}")