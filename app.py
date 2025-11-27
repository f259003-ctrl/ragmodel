# app.py
import streamlit as st
import pandas as pd
import sys
import os
import time

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(__file__))

try:
    from medical_rag_system import MedicalRAGPipeline, initialize_system
except ImportError as e:
    st.error(f"Import error: {e}. Please make sure all required files are present.")

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'last_question' not in st.session_state:
        st.session_state.last_question = ""

def main():
    st.set_page_config(
        page_title="Medical Assistant RAG System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🏥 Medical Assistant RAG System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Important Disclaimer:</strong> This is a demonstration system for educational purposes only. 
    Always consult qualified healthcare professionals for medical advice and treatment.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.title("🔧 Configuration")
        
        gemini_api_key = st.text_input(
            "Enter Gemini API Key:",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        
        st.markdown("---")
        st.subheader("Search Settings")
        max_docs = st.slider("Number of source documents", 1, 10, 3)
        
        st.markdown("---")
        st.subheader("About")
        st.info("""
        This system uses:
        - Medical transcription data
        - FAISS for vector search
        - Gemini AI for responses
        - RAG architecture for accuracy
        """)

    # Initialize system if API key is provided
    if gemini_api_key and not st.session_state.initialized:
        with st.spinner("🚀 Initializing Medical Assistant... This may take a moment."):
            try:
                st.session_state.rag_pipeline = initialize_system(gemini_api_key)
                st.session_state.initialized = True
                st.success("✅ System initialized successfully!")
                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error initializing system: {e}")

    # Main interface
    if st.session_state.initialized:
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask Question", "📋 Sample Queries", "📊 Evaluation", "ℹ️ About"])

        with tab1:
            display_question_interface(max_docs)

        with tab2:
            display_sample_queries()

        with tab3:
            display_evaluation_interface()

        with tab4:
            display_about_section()

    elif gemini_api_key:
        st.warning("🔄 System is initializing... Please wait.")
    else:
        display_getting_started()

def display_question_interface(max_docs):
    """Display the main question-answer interface"""
    st.header("💬 Ask a Medical Question")
    
    # Use last question from session state if available
    question = st.text_area(
        "Enter your medical question:",
        value=st.session_state.last_question,
        placeholder="e.g., What are the common symptoms of diabetes? How is pneumonia treated?",
        height=100,
        key="question_input"
    )

    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("🔍 Get Answer", type="primary", use_container_width=True):
            if question.strip():
                process_question(question, max_docs)
            else:
                st.warning("Please enter a question.")
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.last_question = ""
            st.rerun()

def process_question(question, max_docs):
    """Process a question and display results"""
    st.session_state.last_question = question
    
    with st.spinner("🔍 Searching medical knowledge base..."):
        try:
            result = st.session_state.rag_pipeline.query(question, max_docs)
            
            # Display answer
            st.subheader("💡 Answer")
            st.markdown(f'<div class="success-box">{result["answer"]}</div>', unsafe_allow_html=True)
            
            # Display sources
            st.subheader("📚 Source Documents")
            for i, doc in enumerate(result["source_documents"]):
                with st.expander(f"📄 Source {i+1} | Specialty: {doc['metadata']['medical_specialty']} | Score: {doc['score']:.4f}"):
                    st.write(doc["text"])
                    st.caption(f"🔖 Transcription ID: {doc['metadata']['transcription_id']}")
                    
        except Exception as e:
            st.error(f"❌ Error processing question: {e}")

def display_sample_queries():
    """Display sample medical queries"""
    st.header("📋 Sample Medical Queries")
    st.info("Click any question below to test the system")
    
    sample_queries = [
        "What are the symptoms of myocardial infarction?",
        "How is pneumonia treated?",
        "What are the common symptoms of diabetes?",
        "How are migraines managed?",
        "What causes chest pain in cardiac patients?",
        "How is hypertension diagnosed and treated?",
        "What are the risk factors for stroke?",
        "Describe the treatment for asthma",
        "What are the symptoms of hypothyroidism?",
        "How is bronchitis different from pneumonia?"
    ]
    
    cols = st.columns(2)
    for idx, query in enumerate(sample_queries):
        with cols[idx % 2]:
            if st.button(f"❓ {query}", key=f"sample_{idx}", use_container_width=True):
                st.session_state.last_question = query
                st.rerun()

def display_evaluation_interface():
    """Display evaluation interface"""
    st.header("📊 System Evaluation")
    
    st.subheader("Test Queries")
    test_queries = [
        "What are the symptoms of myocardial infarction?",
        "How is bronchitis treated?",
        "What are the treatment options for diabetes?",
        "Describe the symptoms of pneumonia",
        "What are the common treatments for migraines?",
        "How is asthma diagnosed?",
        "What are the risk factors for heart disease?",
        "Describe hypertension management"
    ]
    
    if st.button("🚀 Run Comprehensive Evaluation", type="primary"):
        run_evaluation(test_queries)
    
    # Quick individual testing
    st.subheader("Quick Test")
    selected_query = st.selectbox("Choose a test query:", test_queries)
    if st.button("Test Selected Query"):
        process_question(selected_query, max_docs=3)

def run_evaluation(test_queries):
    """Run evaluation on test queries"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    total_queries = len(test_queries)
    
    for idx, query in enumerate(test_queries):
        status_text.text(f"Processing: {query}")
        progress_bar.progress((idx + 1) / total_queries)
        
        try:
            result = st.session_state.rag_pipeline.query(query, max_docs=3)
            results.append({
                'Query': query,
                'Answer Preview': result['answer'][:150] + '...' if len(result['answer']) > 150 else result['answer'],
                'Sources Found': len(result['source_documents']),
                'Specialties': ', '.join([doc['metadata']['medical_specialty'] for doc in result['source_documents']]),
                'Avg Score': np.mean([doc['score'] for doc in result['source_documents']]) if result['source_documents'] else 0
            })
        except Exception as e:
            results.append({
                'Query': query,
                'Answer Preview': f'Error: {e}',
                'Sources Found': 0,
                'Specialties': 'None',
                'Avg Score': 0
            })
    
    status_text.text("✅ Evaluation complete!")
    
    # Display results
    df = pd.DataFrame(results)
    st.subheader("Evaluation Results")
    st.dataframe(df, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Queries", len(results))
    with col2:
        successful = len([r for r in results if r['Sources Found'] > 0])
        st.metric("Successful Retrievals", successful)
    with col3:
        success_rate = (successful / len(results)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")

def display_about_section():
    """Display about and information section"""
    st.header("ℹ️ About This System")
    
    st.subheader("System Architecture")
    st.image("https://i.imgur.com/xyz123.png", caption="RAG System Architecture")  # Replace with actual diagram
    
    st.markdown("""
    ### 🔧 Technical Stack
    - **Vector Database**: FAISS for efficient similarity search
    - **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
    - **LLM**: Google Gemini Pro for response generation
    - **Framework**: Streamlit for web interface
    - **Chunking**: Medical context-aware text splitting
    
    ### 📊 Data Sources
    - Medical transcription samples
    - Various medical specialties
    - Evidence-based medical information
    
    ### 🛡️ Safety Features
    - Context-grounded responses only
    - No medical advice generation
    - Clear disclaimers
    - Source attribution
    """)
    
    st.subheader("🚀 Deployment Information")
    st.code("""
    # Local development
    streamlit run app.py
    
    # Deploy to Streamlit Community Cloud
    1. Push code to GitHub
    2. Connect repo at share.streamlit.io
    3. Set Gemini API key as secrets
    """)

def display_getting_started():
    """Display getting started instructions"""
    st.header("🚀 Getting Started")
    
    st.markdown("""
    ### Welcome to the Medical Assistant RAG System!
    
    To get started:
    
    1. **Get a Gemini API Key**:
       - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
       - Create a free account if you don't have one
       - Generate an API key
    
    2. **Enter Your API Key**:
       - Look at the sidebar on the left
       - Paste your Gemini API key in the input field
       - The system will initialize automatically
    
    3. **Start Asking Questions**:
       - Use the "Ask Question" tab for custom queries
       - Try sample questions from the "Sample Queries" tab
       - Run evaluations to test system performance
    
    ### 🔒 Privacy & Security
    - Your API key is stored only in your session
    - No data is permanently stored
    - All processing happens in real-time
    """)
    
    st.info("💡 **Tip**: Start with the sample queries to see how the system works!")

if __name__ == "__main__":
    main()
