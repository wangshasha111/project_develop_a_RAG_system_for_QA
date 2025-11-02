"""
Multimodal RAG System for AI Research Paper Question Answering
A comprehensive Streamlit-based UI for querying AI research papers using RAG
"""

import streamlit as st
import os
from pathlib import Path
from typing import Optional
import base64
from io import BytesIO
from PIL import Image as PILImage
import streamlit.components.v1 as components
import pyperclip

# Import custom modules
try:
    from document_processor import DocumentProcessor, get_mock_documents
    from retriever import MultimodalRetriever, get_mock_retriever
    from rag_chain import RAGChain, get_mock_rag_chain
    from config import Config
    from utils import get_mock_response, encode_image_for_display
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()


def get_api_key(key_name: str) -> Optional[str]:
    """
    Get API key from Streamlit secrets (cloud) or environment variables (local).
    
    This function supports both Streamlit Cloud deployment and local development.
    Priority: Streamlit secrets > Environment variables
    
    Args:
        key_name: Name of the API key (e.g., 'OPENAI_API_KEY')
        
    Returns:
        API key string if found, None otherwise
    """
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except (FileNotFoundError, KeyError):
        pass
    
    # Fall back to environment variables (for local development)
    return os.getenv(key_name)

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .prompt-button {
        margin: 5px 0;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .source-section {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #fafafa;
        border-radius: 0.5rem;
        border-left: 3px solid #1976d2;
    }
    .source-content {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.3rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        font-family: monospace;
        white-space: pre-wrap;
        word-wrap: break-word;
        max-height: 300px;
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.chat_history = []
    st.session_state.retriever = None
    st.session_state.rag_chain = None
    st.session_state.debug_mode = True
    st.session_state.api_keys_set = False
    st.session_state.preprocessing_done = False
    st.session_state.auto_load_attempted = False

# Default prompt templates
PROMPT_TEMPLATES = {
    "RAG Components": "What are the main components of a RAG model, and how do they interact?",
    "Transformer Layers": "What are the two sub-layers in each encoder layer of the Transformer model?",
    "Positional Encoding": "Explain how positional encoding is implemented in Transformers and why it is necessary.",
    "Multi-head Attention": "Describe the concept of multi-head attention in the Transformer architecture. Why is it beneficial?",
    "Few-shot Learning": "What is few-shot learning, and how does GPT-3 implement it during inference?",
    "Custom": ""  # For user's custom prompt
}

def check_preprocessing_status():
    """Check if preprocessing has been completed"""
    import json
    metadata_file = Path('.preprocessing_metadata.json')
    
    if not metadata_file.exists():
        return False, None
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if metadata.get('status') == 'ready':
            return True, metadata
    except Exception:
        pass
    
    return False, None

def load_preprocessed_data():
    """Load retriever and RAG chain from preprocessed data"""
    try:
        # Get provider and model from metadata
        is_ready, metadata = check_preprocessing_status()
        
        if not is_ready:
            return None, "Preprocessing not completed. Please run preprocess.py first."
        
        provider = metadata.get('provider', 'OpenAI')
        model = metadata.get('model', 'gpt-4o-mini')
        
        # Set API keys from environment or Streamlit secrets
        if provider == 'OpenAI':
            api_key = get_api_key('OPENAI_API_KEY')
            if not api_key:
                return None, "OpenAI API key not found in environment or secrets"
            os.environ['OPENAI_API_KEY'] = api_key
        elif provider == 'Google':
            api_key = get_api_key('GOOGLE_API_KEY')
            if not api_key:
                return None, "Google API key not found in environment or secrets"
            os.environ['GOOGLE_API_KEY'] = api_key
        
        # Load retriever from existing storage
        retriever = MultimodalRetriever(provider=provider, model=model)
        
        # Load from ChromaDB and Redis
        chroma_dir = metadata.get('chroma_directory', 'chroma_db')
        retriever.load_from_storage(persist_directory=chroma_dir)
        
        # Create RAG chain
        rag_chain = RAGChain(
            retriever=retriever,
            provider=provider,
            model=model
        )
        
        return {
            'retriever': retriever,
            'rag_chain': rag_chain,
            'provider': provider,
            'model': model,
            'metadata': metadata
        }, None
        
    except Exception as e:
        import traceback
        error_msg = f"Error loading preprocessed data: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg

def initialize_system():
    """Initialize the RAG system - DEPRECATED, use auto-load instead"""
    st.warning("⚠️ Manual initialization is deprecated. The system now loads preprocessed data automatically.")
    st.info("ℹ️ If you need to reprocess documents, please run: python preprocess.py")
    return False

def render_sidebar():
    """Render the sidebar with configuration options"""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Debug Mode
        st.subheader("🐛 Debug Mode")
        debug_mode = st.toggle(
            "Enable Debug Mode",
            value=st.session_state.debug_mode,
            help="Use mock data to test the UI without API calls"
        )
        st.session_state.debug_mode = debug_mode
        
        if debug_mode:
            st.info("📝 Debug mode is ON - No API calls will be made")
        
        st.divider()
        
        # Provider Selection
        st.subheader("🤖 AI Provider")
        provider = st.selectbox(
            "Select Provider",
            options=["OpenAI", "Google"],
            index=0,
            disabled=debug_mode
        )
        st.session_state.selected_provider = provider
        
        # Model Selection based on provider
        if provider == "OpenAI":
            model_options = ["gpt-4o", "gpt-4o-mini"]
            default_model = "gpt-4o"
        else:  # Google
            model_options = ["gemini-2.0-flash-exp", "gemini-1.5-pro"]
            default_model = "gemini-2.0-flash-exp"
        
        model = st.selectbox(
            "Select Model",
            options=model_options,
            index=0,
            disabled=debug_mode
        )
        st.session_state.selected_model = model
        
        st.divider()
        
        # API Key Input
        st.subheader("🔑 API Keys")
        
        if not debug_mode:
            # Check if API keys are available from Streamlit secrets
            has_secret_key = False
            
            if provider == "OpenAI":
                secret_key = get_api_key('OPENAI_API_KEY')
                if secret_key:
                    st.success("✅ OpenAI API key loaded from secrets")
                    has_secret_key = True
                    st.session_state.openai_api_key = secret_key
                    st.session_state.api_keys_set = True
                else:
                    openai_key = st.text_input(
                        "OpenAI API Key",
                        type="password",
                        value=st.session_state.get('openai_api_key', ''),
                        help="Get your API key from https://platform.openai.com/api-keys"
                    )
                    if openai_key:
                        st.session_state.openai_api_key = openai_key
                        st.session_state.api_keys_set = True
                    else:
                        st.session_state.api_keys_set = False
                    
            else:  # Google
                secret_key = get_api_key('GOOGLE_API_KEY')
                if secret_key:
                    st.success("✅ Google API key loaded from secrets")
                    has_secret_key = True
                    st.session_state.google_api_key = secret_key
                    st.session_state.api_keys_set = True
                else:
                    google_key = st.text_input(
                        "Google API Key",
                        type="password",
                        value=st.session_state.get('google_api_key', ''),
                        help="Get your API key from https://makersuite.google.com/app/apikey"
                    )
                    if google_key:
                        st.session_state.google_api_key = google_key
                        st.session_state.api_keys_set = True
                    else:
                        st.session_state.api_keys_set = False
        else:
            st.info("✨ API keys not needed in debug mode - using high-quality mock data")
        
        st.divider()
        
        # System Status
        st.subheader("📊 System Status")
        
        # Check preprocessing status
        is_ready, metadata = check_preprocessing_status()
        
        if st.session_state.initialized:
            st.success("✅ System Ready")
            
            # Show system info
            if metadata:
                st.info(f"""
                **Loaded System:**
                - Provider: {metadata.get('provider', 'N/A')}
                - Model: {metadata.get('model', 'N/A')}
                - Preprocessed: {metadata.get('preprocessing_date', 'N/A')[:10]}
                """)
            
            if st.button("🔄 Reload System"):
                st.session_state.auto_load_attempted = False
                st.rerun()
        else:
            st.warning("⚠️ System Not Ready")
            
            if is_ready:
                st.info("Preprocessed data found. The system will auto-load on next refresh.")
                if st.button("� Load Now"):
                    st.session_state.auto_load_attempted = False
                    st.rerun()
            else:
                st.error("""
                **Preprocessing Required**
                
                Run in terminal:
                ```bash
                python preprocess.py
                ```
                """)
        
        st.divider()
        
        # Clear Chat History
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Additional Info
        st.divider()
        st.subheader("ℹ️ About")
        st.info("""
        This is a Multimodal RAG system that can answer questions about AI research papers.
        
        **Features:**
        - Multi-modal retrieval (text, tables, images)
        - Source attribution
        - Chat history
        - Multiple AI providers
        """)

def render_prompt_buttons():
    """Render predefined prompt template buttons"""
    st.subheader("📝 Quick Prompts")
    
    cols = st.columns(3)
    
    for idx, (name, prompt) in enumerate(PROMPT_TEMPLATES.items()):
        if name != "Custom":
            with cols[idx % 3]:
                if st.button(name, key=f"prompt_{idx}", use_container_width=True):
                    st.session_state.current_prompt = prompt
                    st.rerun()

def render_chat_interface():
    """Render the main chat interface"""
    st.title("📚 Multimodal RAG System for AI Research Papers")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for idx, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:  # assistant
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Copy button for assistant response with working functionality
                copy_col1, copy_col2 = st.columns([1, 5])
                with copy_col1:
                    # Use unique key for each copy button
                    button_key = f"copy_btn_{idx}"
                    if st.button("📋 Copy", key=button_key, help="Copy response to clipboard"):
                        try:
                            # Use pyperclip to copy to clipboard
                            pyperclip.copy(message['content'])
                            st.session_state[f'copied_{idx}'] = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to copy: {str(e)}")
                
                with copy_col2:
                    if st.session_state.get(f'copied_{idx}', False):
                        st.success("✅ Copied to clipboard!")
                        # Reset after showing
                        if st.button("Clear", key=f"clear_{idx}"):
                            st.session_state[f'copied_{idx}'] = False
                            st.rerun()
                
                # Display sources if available with DETAILED content
                if 'sources' in message and message['sources']:
                    with st.expander("📎 View Detailed Sources", expanded=False):
                        sources = message['sources']
                        
                        # Display text sources with full content AND summaries
                        if sources.get('texts'):
                            st.markdown("### 📄 Text Sources")
                            st.info(f"Found {len(sources['texts'])} relevant text passages")
                            
                            for text_idx, text_data in enumerate(sources['texts'], 1):
                                st.markdown(f"#### Source {text_idx}")
                                
                                # Handle both dict format (new) and string format (old)
                                if isinstance(text_data, dict):
                                    text_content = text_data.get('content', text_data)
                                    text_summary = text_data.get('summary', '')
                                else:
                                    text_content = text_data
                                    text_summary = ''
                                
                                # Decode bytes to string if needed
                                if isinstance(text_content, bytes):
                                    text_content = text_content.decode('utf-8')
                                
                                # DEBUG: Always show both summary and content
                                st.markdown("**📝 AI-Generated Summary:**")
                                if text_summary:
                                    st.info(text_summary)
                                else:
                                    st.warning("⚠️ No summary available (this is expected if doc_id_mapping.json doesn't exist)")
                                
                                # Display original content
                                st.markdown("**📄 Original Content:**")
                                st.markdown(f'<div class="source-content">{text_content}</div>', 
                                          unsafe_allow_html=True)
                                
                                # Add copy buttons
                                col1, col2, col3 = st.columns([1, 1, 4])
                                
                                with col1:
                                    copy_content_key = f"copy_content_{idx}_{text_idx}"
                                    if st.button("📋 Copy Content", key=copy_content_key, help="Copy original content"):
                                        try:
                                            pyperclip.copy(text_content)
                                            st.session_state[f"content_copied_{idx}_{text_idx}"] = True
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ Failed to copy: {str(e)}")
                                
                                with col2:
                                    if text_summary:
                                        copy_summary_key = f"copy_summary_{idx}_{text_idx}"
                                        if st.button("📋 Copy Summary", key=copy_summary_key, help="Copy AI summary"):
                                            try:
                                                pyperclip.copy(text_summary)
                                                st.session_state[f"summary_copied_{idx}_{text_idx}"] = True
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"❌ Failed to copy: {str(e)}")
                                
                                with col3:
                                    if st.session_state.get(f"content_copied_{idx}_{text_idx}", False):
                                        st.success("✅ Content Copied!")
                                    if st.session_state.get(f"summary_copied_{idx}_{text_idx}", False):
                                        st.success("✅ Summary Copied!")
                                
                                st.divider()
                        
                        # Display table sources with summaries
                        if sources.get('tables'):
                            st.markdown("### 📊 Table Sources")
                            st.info(f"Found {len(sources.get('tables', []))} relevant tables")
                            
                            for table_idx, table_data in enumerate(sources.get('tables', []), 1):
                                st.markdown(f"#### Table {table_idx}")
                                
                                # Handle both dict format (new) and string format (old)
                                if isinstance(table_data, dict):
                                    table_content = table_data.get('content', table_data)
                                    table_summary = table_data.get('summary', '')
                                else:
                                    table_content = table_data
                                    table_summary = ''
                                
                                # Decode bytes to string if needed
                                if isinstance(table_content, bytes):
                                    table_content = table_content.decode('utf-8')
                                
                                # Display summary
                                st.markdown("**📝 AI-Generated Summary:**")
                                if table_summary:
                                    st.info(table_summary)
                                else:
                                    st.warning("⚠️ No summary available")
                                
                                # Display table content
                                st.markdown("**📊 Table Content:**")
                                st.markdown(table_content)
                                
                                # Add copy buttons
                                col1, col2, col3 = st.columns([1, 1, 4])
                                
                                with col1:
                                    copy_table_key = f"copy_table_{idx}_{table_idx}"
                                    if st.button("📋 Copy Table", key=copy_table_key, help="Copy table content"):
                                        try:
                                            pyperclip.copy(table_content)
                                            st.session_state[f"table_copied_{idx}_{table_idx}"] = True
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ Failed to copy: {str(e)}")
                                
                                with col2:
                                    if table_summary:
                                        copy_table_summary_key = f"copy_table_summary_{idx}_{table_idx}"
                                        if st.button("📋 Copy Summary", key=copy_table_summary_key, help="Copy table summary"):
                                            try:
                                                pyperclip.copy(table_summary)
                                                st.session_state[f"table_summary_copied_{idx}_{table_idx}"] = True
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"❌ Failed to copy: {str(e)}")
                                
                                with col3:
                                    if st.session_state.get(f"table_copied_{idx}_{table_idx}", False):
                                        st.success("✅ Table Copied!")
                                    if st.session_state.get(f"table_summary_copied_{idx}_{table_idx}", False):
                                        st.success("✅ Summary Copied!")
                                
                                st.divider()
                        
                        # Display images with better layout and summaries
                        if sources.get('images'):
                            st.markdown("### 🖼️ Image Sources")
                            st.info(f"Found {len(sources['images'])} relevant images")
                            
                            # Display each image with its summary
                            for img_idx, img_data in enumerate(sources['images'], 1):
                                st.markdown(f"#### Image {img_idx}")
                                
                                # Handle both dict format (new) and string format (old)
                                if isinstance(img_data, dict):
                                    img_content = img_data.get('content', img_data)
                                    img_summary = img_data.get('summary', '')
                                else:
                                    img_content = img_data
                                    img_summary = ''
                                
                                # Display summary
                                st.markdown("**📝 AI-Generated Image Description:**")
                                if img_summary:
                                    st.info(img_summary)
                                else:
                                    st.warning("⚠️ No image description available")
                                
                                # Display image
                                try:
                                    img_decoded = base64.b64decode(img_content)
                                    img = PILImage.open(BytesIO(img_decoded))
                                    st.image(img, 
                                           caption=f"Figure {img_idx}", 
                                           use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error displaying image {img_idx}: {str(e)}")
                                
                                # Add copy summary button if available
                                if img_summary:
                                    copy_img_summary_key = f"copy_img_summary_{idx}_{img_idx}"
                                    if st.button("📋 Copy Image Description", 
                                               key=copy_img_summary_key, 
                                               help="Copy AI-generated image description"):
                                        try:
                                            pyperclip.copy(img_summary)
                                            st.session_state[f"img_summary_copied_{idx}_{img_idx}"] = True
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ Failed to copy: {str(e)}")
                                    
                                    if st.session_state.get(f"img_summary_copied_{idx}_{img_idx}", False):
                                        st.success("✅ Description Copied!")
                                
                                st.divider()
        
        st.divider()
    
    # Prompt buttons
    render_prompt_buttons()
    
    st.divider()
    
    # Question input
    st.subheader("❓ Ask a Question")
    
    # Get default prompt
    default_prompt = st.session_state.get('current_prompt', PROMPT_TEMPLATES["RAG Components"])
    
    question = st.text_area(
        "Enter your question:",
        value=default_prompt,
        height=100,
        placeholder="Type your question here or use one of the quick prompts above..."
    )
    
    # Send button
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        send_button = st.button("🚀 Send Question", type="primary", use_container_width=True)
    
    with col2:
        clear_input = st.button("🔄 Clear Input", use_container_width=True)
    
    if clear_input:
        st.session_state.current_prompt = ""
        st.rerun()
    
    # Process question
    if send_button and question.strip():
        if not st.session_state.initialized and not st.session_state.debug_mode:
            st.warning("⚠️ Please initialize the system first using the sidebar button")
            return
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': question
        })
        
        # Generate response
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                if st.session_state.debug_mode:
                    # Use mock response
                    response = get_mock_response(question)
                else:
                    # Use actual RAG chain
                    response = st.session_state.rag_chain.query(
                        question,
                        chat_history=st.session_state.chat_history[:-1]
                    )
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response['answer'],
                    'sources': response.get('sources', {})
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                st.session_state.chat_history.pop()  # Remove user message on error

def main():
    """Main application entry point"""
    # Auto-load preprocessed data on first run
    if not st.session_state.auto_load_attempted:
        st.session_state.auto_load_attempted = True
        
        # Check if debug mode
        if st.session_state.debug_mode:
            # Load mock data
            st.session_state.rag_chain = get_mock_rag_chain()
            st.session_state.initialized = True
            st.session_state.preprocessing_done = True
        else:
            # Try to load preprocessed data
            is_ready, metadata = check_preprocessing_status()
            
            if is_ready:
                with st.spinner("🔄 Loading preprocessed RAG system..."):
                    result, error = load_preprocessed_data()
                    
                    if result:
                        st.session_state.retriever = result['retriever']
                        st.session_state.rag_chain = result['rag_chain']
                        st.session_state.initialized = True
                        st.session_state.preprocessing_done = True
                        
                        # Show success message
                        st.success(f"""
                        ✅ **RAG System Loaded Successfully!**
                        
                        - Provider: {result['provider']}
                        - Model: {result['model']}
                        - Preprocessed: {result['metadata']['preprocessing_date']}
                        
                        You can now ask questions about the documents!
                        """)
                    else:
                        st.error(f"❌ Failed to load preprocessed data: {error}")
                        st.warning("""
                        ⚠️ **Preprocessing Required**
                        
                        Please run the preprocessing script first:
                        ```bash
                        python preprocess.py
                        ```
                        
                        Or enable Debug Mode in the sidebar to test without real data.
                        """)
            else:
                st.warning("""
                ⚠️ **No Preprocessed Data Found**
                
                Please run the preprocessing script to prepare the RAG system:
                ```bash
                python preprocess.py
                ```
                
                This will:
                1. Process all PDF documents
                2. Generate summaries and embeddings  
                3. Store data in ChromaDB and Redis
                
                **Or** enable Debug Mode in the sidebar to test the UI with mock data.
                """)
    
    render_sidebar()
    render_chat_interface()

if __name__ == "__main__":
    main()
