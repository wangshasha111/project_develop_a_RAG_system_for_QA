#!/usr/bin/env python3
"""
Setup Verification Script
Checks if all dependencies and requirements are properly installed
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_warning(text):
    """Print warning message"""
    print(f"⚠️  {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def check_python_version():
    """Check if Python version is adequate"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print_success("Python version is compatible")
        return True
    else:
        print_error("Python 3.8 or higher is required")
        return False

def check_package_installed(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print_success(f"{package_name} is installed")
        return True
    except ImportError:
        print_error(f"{package_name} is NOT installed")
        return False

def check_core_dependencies():
    """Check core Python dependencies"""
    print_header("Checking Core Dependencies")
    
    packages = [
        ("streamlit", "streamlit"),
        ("langchain", "langchain"),
        ("langchain-openai", "langchain_openai"),
        ("langchain-community", "langchain_community"),
        ("langchain-chroma", "langchain_chroma"),
        ("unstructured", "unstructured"),
        ("chromadb", "chromadb"),
        ("redis", "redis"),
        ("PIL", "PIL"),
        ("nltk", "nltk"),
    ]
    
    results = []
    for package_name, import_name in packages:
        results.append(check_package_installed(package_name, import_name))
    
    return all(results)

def check_optional_dependencies():
    """Check optional dependencies"""
    print_header("Checking Optional Dependencies")
    
    # Check for Redis server
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, socket_timeout=1)
        client.ping()
        print_success("Redis server is running")
    except Exception as e:
        print_warning("Redis server is not running (optional - debug mode will work without it)")
    
    # Check for system dependencies
    print("\nSystem Dependencies (optional for better PDF processing):")
    
    # Check Tesseract
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, 
                              timeout=5)
        if result.returncode == 0:
            print_success("Tesseract OCR is installed")
        else:
            print_warning("Tesseract OCR not found (optional)")
    except:
        print_warning("Tesseract OCR not found (optional)")
    
    # Check Poppler (pdftoimage)
    try:
        result = subprocess.run(['pdftoimage', '-v'], 
                              capture_output=True, 
                              timeout=5)
        if result.returncode == 0:
            print_success("Poppler (pdftoimage) is installed")
        else:
            print_warning("Poppler not found (optional)")
    except:
        print_warning("Poppler not found (optional)")

def check_project_structure():
    """Check if project structure is correct"""
    print_header("Checking Project Structure")
    
    base_dir = Path(__file__).parent
    
    required_files = [
        "app.py",
        "document_processor.py",
        "retriever.py",
        "rag_chain.py",
        "config.py",
        "utils.py",
        "requirements.txt",
        "README.md",
        "README_CN.md",
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = base_dir / filename
        if filepath.exists():
            print_success(f"{filename} exists")
        else:
            print_error(f"{filename} is missing")
            all_exist = False
    
    # Check dataset folder
    dataset_dir = base_dir / "RAG Project Dataset"
    if dataset_dir.exists():
        pdf_files = list(dataset_dir.glob("*.pdf"))
        print_success(f"Dataset folder exists with {len(pdf_files)} PDF files")
    else:
        print_error("Dataset folder is missing")
        all_exist = False
    
    return all_exist

def download_nltk_data():
    """Download required NLTK data"""
    print_header("Downloading NLTK Data")
    
    try:
        import nltk
        
        datasets = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger']
        for dataset in datasets:
            try:
                nltk.data.find(f'tokenizers/{dataset}')
                print_success(f"{dataset} already downloaded")
            except LookupError:
                print(f"Downloading {dataset}...")
                nltk.download(dataset, quiet=True)
                print_success(f"{dataset} downloaded")
        
        return True
    except Exception as e:
        print_error(f"Error downloading NLTK data: {e}")
        return False

def main():
    """Main verification function"""
    print("\n" + "🔍" * 30)
    print("  RAG System Setup Verification")
    print("🔍" * 30)
    
    results = []
    
    # Check Python version
    results.append(check_python_version())
    
    # Check project structure
    results.append(check_project_structure())
    
    # Check core dependencies
    results.append(check_core_dependencies())
    
    # Check optional dependencies
    check_optional_dependencies()
    
    # Download NLTK data
    results.append(download_nltk_data())
    
    # Final summary
    print_header("Verification Summary")
    
    if all(results):
        print_success("All critical checks passed!")
        print("\n✨ Your system is ready to run the RAG application!")
        print("\nTo start the application:")
        print("  1. macOS: Double-click 'run.command'")
        print("  2. Or run: streamlit run app.py")
        return 0
    else:
        print_error("Some checks failed!")
        print("\n📝 Please fix the issues above before running the application.")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
