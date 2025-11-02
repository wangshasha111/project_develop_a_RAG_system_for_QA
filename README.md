# 🤖 Multimodal RAG System for AI Research Papers

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Retrieval-Augmented Generation (RAG) system with an interactive Streamlit interface for querying AI research papers. Supports multimodal content including text, tables, and images.

![RAG System Demo](https://via.placeholder.com/800x400?text=RAG+System+Interface)

## ✨ Features

- 🔍 **Advanced Retrieval**: Multi-vector retriever with semantic search
- 🤖 **Multiple AI Providers**: OpenAI (GPT-4o, GPT-4o-mini) and Google (Gemini 2.0, 1.5 Pro)
- 📊 **Multimodal Support**: Process text, tables, and images from PDFs
- 💬 **Chat Interface**: Conversational Q&A with context awareness
- 🐛 **Debug Mode**: Test without API keys using high-quality mock data
- 📎 **Source Attribution**: Every answer includes references to source documents
- 🎨 **Modern UI**: Clean, responsive Streamlit interface

## 🚀 Quick Start

### Option 1: Try Debug Mode (No Setup Required)

1. Click the Streamlit badge above to visit the live app
2. Toggle "Enable Debug Mode" in the sidebar
3. Click "🚀 Initialize System"
4. Start asking questions immediately!

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/multimodal-rag-system.git
cd multimodal-rag-system

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger')"

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🔑 API Keys

### For OpenAI (GPT models):
1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Enter it in the sidebar when selecting OpenAI as provider

### For Google (Gemini models):
1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Enter it in the sidebar when selecting Google as provider

**Note**: Debug mode works perfectly without any API keys!

## 📖 Usage

### Debug Mode (Recommended for First Time)

1. Launch the application
2. Toggle "Enable Debug Mode" in the sidebar
3. Click "🚀 Initialize System"
4. Try the quick prompt buttons or ask custom questions
5. Explore the detailed, pre-configured responses

### Production Mode (With API Keys)

1. Select AI provider (OpenAI or Google)
2. Choose your preferred model
3. Enter your API key
4. Click "🚀 Initialize System" (processes research papers, takes 2-5 minutes)
5. Start asking questions about AI research papers

### Example Questions

Try these questions in debug mode:

- **"What are the main components of a RAG model?"**
- **"How does positional encoding work in Transformers?"**
- **"Explain multi-head attention and its benefits"**
- **"What is few-shot learning in GPT-3?"**
- **"Describe the training objectives used in Transformers"**

## 🏗️ Architecture

```
User Query
    ↓
[Document Processor]
├─ PDF Parsing
├─ Text Extraction
├─ Table Extraction  
└─ Image Extraction
    ↓
[Multi-Vector Retriever]
├─ Generate Summaries
├─ Create Embeddings
├─ Vector DB (Chroma)
└─ Document Store (Redis/Memory)
    ↓
[RAG Chain]
├─ Assemble Context
├─ Construct Prompt
└─ Generate Answer
    ↓
Answer + Sources
```

## 📁 Project Structure

```
multimodal-rag-system/
├── app.py                 # Main Streamlit application
├── document_processor.py  # PDF processing and extraction
├── retriever.py          # Multi-vector retriever
├── rag_chain.py          # RAG pipeline
├── config.py             # Configuration settings
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── packages.txt          # System dependencies (for Streamlit Cloud)
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── README.md             # This file
```

## 🌐 Deployment to Streamlit Cloud

### Prerequisites

1. Fork or clone this repository to your GitHub account
2. Sign up for [Streamlit Cloud](https://share.streamlit.io/)

### Steps

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Click "New app"
3. Select your repository
4. Set main file path: `app.py`
5. Click "Advanced settings" to add secrets (optional):

```toml
# Add these in Streamlit Cloud Secrets (if using production mode)
OPENAI_API_KEY = "your-openai-key-here"
GOOGLE_API_KEY = "your-google-key-here"
```

6. Click "Deploy!"

**Note**: The app works great in debug mode without any secrets configured!

## 🎓 Research Papers Included

The system is pre-configured to query these foundational AI papers:

1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Original Transformer architecture
   - Self-attention mechanisms
   - Positional encoding

2. **"Language Models are Few-Shot Learners"** (Brown et al., 2020)
   - GPT-3 architecture and training
   - Few-shot learning capabilities
   - Scaling laws

3. **"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** (Lewis et al., 2020)
   - RAG architecture
   - Retrieval methods
   - Generation approaches

## 🛠️ Development

### Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- Internet connection (for API calls in production mode)

### Optional Dependencies

For enhanced PDF processing:

```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian  
sudo apt-get install tesseract-ocr poppler-utils

# Windows
# Download from official websites
```

For persistent storage:

```bash
# macOS
brew install redis
brew services start redis
```

### Running Tests

```bash
# Test retriever
python test_retriever.py

# Test full chain
python test_full_chain.py

# Verify setup
python verify_setup.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Unstructured**: Document parsing
- **LangChain**: RAG framework
- **Streamlit**: Web interface
- **OpenAI & Google**: AI models
- **Chroma**: Vector database

## 📞 Support

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/YOUR_USERNAME/multimodal-rag-system/issues)
- 📖 Documentation: See [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)

## ⭐ Star History

If you find this project helpful, please consider giving it a star!

---

**Built with ❤️ using Streamlit and LangChain**
