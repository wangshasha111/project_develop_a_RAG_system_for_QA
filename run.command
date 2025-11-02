#!/bin/bash

# Navigate to the directory containing this script
cd "$(dirname "$0")"

echo "=================================================="
echo "  Multimodal RAG System Launcher"
echo "=================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    echo "   Visit: https://www.python.org/downloads/"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Virtual environment not found. Creating one..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment."
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "✅ Virtual environment created successfully!"
    echo ""
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if [ ! -f "venv/bin/streamlit" ]; then
    echo "📥 Installing dependencies... This may take a few minutes..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies."
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "✅ Dependencies installed successfully!"
    echo ""
fi

# Download required NLTK data
echo "📚 Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"
echo ""

# Check if Redis is needed (for non-debug mode)
echo "ℹ️  Note: Redis is optional. The system will work in debug mode without it."
echo "   To use full functionality, install Redis:"
echo "   brew install redis"
echo "   brew services start redis"
echo ""

# Launch Streamlit app
echo "🚀 Launching Multimodal RAG System..."
echo ""
echo "=================================================="
echo "  The app will open in your default browser"
echo "  Press Ctrl+C in this window to stop the server"
echo "=================================================="
echo ""

streamlit run app.py

# Deactivate virtual environment on exit
deactivate
