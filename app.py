import streamlit as st

# Page config - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Hybrid RAG System", 
    layout="wide",
    page_icon="📊"
)

import os
import tempfile
from pathlib import Path
from main_dynamic import process_uploaded_files, ask_question, clear_session

# Initialize session state
if 'files_processed' not in st.session_state:
    st.session_state.files_processed = False
if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = []

st.title("📊 Hybrid RAG + Data Agent System")
st.markdown("**Upload your documents (PDF, TXT, CSV) and ask questions!**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # File upload section
    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt', 'csv'],
        accept_multiple_files=True,
        help="Upload PDF, TXT, or CSV files to query"
    )
    
    # Process files button
    if uploaded_files:
        if st.button("🔄 Process Files", type="primary"):
            with st.spinner("Processing your files... This may take a moment."):
                try:
                    # Clear previous session
                    clear_session()
                    
                    # Save uploaded files temporarily
                    temp_dir = tempfile.mkdtemp()
                    saved_files = []
                    
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        saved_files.append(file_path)
                    
                    # Process the files
                    result = process_uploaded_files(saved_files)
                    
                    st.session_state.files_processed = True
                    st.session_state.uploaded_file_names = [f.name for f in uploaded_files]
                    
                    st.success(f"✅ Processed {result['total_docs']} documents!")
                    st.info(f"📄 PDFs: {result['pdf_count']} | 📝 Text: {result['text_count']} | 📊 CSV: {result['csv_count']}")
                    
                except Exception as e:
                    st.error(f"❌ Error processing files: {str(e)}")
                    st.session_state.files_processed = False
    
    # Show processed files
    if st.session_state.files_processed:
        st.success("✅ Files Ready")
        with st.expander("📋 Uploaded Files"):
            for fname in st.session_state.uploaded_file_names:
                st.text(f"• {fname}")
        
        if st.button("🗑️ Clear All"):
            clear_session()
            st.session_state.files_processed = False
            st.session_state.uploaded_file_names = []
            st.rerun()
    
    st.divider()
    
    # Display options
    show_context = st.checkbox("Show Retrieved Context", value=False)
    show_sources = st.checkbox("Show Sources", value=True)
    
    st.divider()
    
    # Example queries
    with st.expander("💡 Example Queries"):
        st.markdown("""
        **Aggregation queries:**
        - What is the average value?
        - How many records are there?
        - What's the total sum?
        
        **Semantic queries:**
        - What is the refund policy?
        - Summarize the document
        - What are the key points?
        """)

# Main content area
if not st.session_state.files_processed:
    # Welcome screen
    st.info("👈 **Get Started:** Upload your documents using the sidebar")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📄 PDF Support")
        st.markdown("Upload PDF documents for semantic search and analysis")
        
    with col2:
        st.markdown("### 📊 CSV Analysis")
        st.markdown("Run statistical queries on your data tables")
        
    with col3:
        st.markdown("### 📝 Text Files")
        st.markdown("Query information from plain text documents")
    
    st.divider()
    
    st.markdown("### 🎯 How It Works")
    st.markdown("""
    1. **Upload** your documents (PDF, TXT, CSV)
    2. **Process** them to build the knowledge base
    3. **Ask** questions in natural language
    4. **Get** intelligent answers with sources
    """)
    
    st.markdown("### ✨ Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 🤖 **Intelligent Query Routing**
        - 🔍 **Semantic Search (RAG)**
        - 📊 **Statistical Analysis (Pandas Agent)**
        """)
    
    with col2:
        st.markdown("""
        - 📚 **Multi-Document Support**
        - 🎯 **Source Attribution**
        - ⚡ **Real-time Processing**
        """)

else:
    # Query interface
    st.markdown("### 🔍 Ask a Question")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the average spending score? or What is mentioned about refunds?",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 New Question", use_container_width=True):
            st.rerun()
    
    if ask_button and query.strip():
        with st.spinner("🤔 Thinking..."):
            try:
                response = ask_question(query)
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.markdown(f"**{response['answer']}**")
                
                # Display sources
                if show_sources and response['sources']:
                    st.markdown("### 📚 Sources")
                    for i, src in enumerate(response['sources'], 1):
                        if isinstance(src, dict):
                            st.markdown(f"**{i}.** {src.get('source', 'Unknown')} ({src.get('type', 'Unknown type')})")
                        else:
                            st.markdown(f"**{i}.** {src}")
                
                # Display context
                if show_context and response.get('context'):
                    with st.expander("🧠 Retrieved Context"):
                        st.text(response['context'])
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Tip: Try rephrasing your question or check if the information exists in your uploaded documents")
    
    elif ask_button and not query.strip():
        st.warning("⚠️ Please enter a question")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    Built with ❤️ using LangChain, OpenAI, FAISS, and Streamlit
</div>
""", unsafe_allow_html=True)
