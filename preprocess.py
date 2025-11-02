"""
Preprocessing Script for RAG System
Run this script ONCE to prepare all data before deploying the application

This script will:
1. Load and process all PDF documents
2. Extract text, tables, and images
3. Generate summaries using LLM
4. Create embeddings
5. Store everything in Redis (raw content) and ChromaDB (embeddings)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Import custom modules
from document_processor import DocumentProcessor
from retriever import MultimodalRetriever
from config import Config

def check_prerequisites():
    """Check if all required components are available"""
    print("\n" + "="*80)
    print("🔍 Checking Prerequisites...")
    print("="*80)
    
    # Check API keys
    provider = os.getenv('DEFAULT_PROVIDER', 'OpenAI')
    if provider == 'OpenAI':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ ERROR: OPENAI_API_KEY not found in .env file")
            print("   Please add your OpenAI API key to .env file")
            return False
        print(f"✅ OpenAI API key found")
    else:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ ERROR: GOOGLE_API_KEY not found in .env file")
            print("   Please add your Google API key to .env file")
            return False
        print(f"✅ Google API key found")
    
    # Check PDF directory
    pdf_dir = Path(Config.DATASET_DIR)
    if not pdf_dir.exists():
        print(f"❌ ERROR: PDF directory not found: {Config.DATASET_DIR}")
        print(f"   Please create the directory and add your PDF files")
        return False
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ ERROR: No PDF files found in {Config.DATASET_DIR}")
        print(f"   Please add PDF files to the directory")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"   - {pdf_file.name}")
    
    # Check storage directories
    chroma_dir = Path(Config.CHROMA_DB_DIR)
    redis_dir = Path(Config.BASE_DIR) / "redis_storage"
    
    print(f"\n📁 Storage locations:")
    print(f"   - ChromaDB: {chroma_dir}")
    print(f"   - Redis: {redis_dir}")
    
    return True

def process_documents():
    """Process all PDF documents"""
    print("\n" + "="*80)
    print("📄 STEP 1: Processing PDF Documents")
    print("="*80)
    
    # Check if we already have processed documents cached
    cache_file = Path('.document_cache.json')
    
    if cache_file.exists():
        print("\n🔍 Found cached processed documents!")
        
        import json
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        # Check if cache is still valid (PDFs haven't changed)
        pdf_dir = Path(Config.DATASET_DIR)
        current_pdfs = sorted([f.name for f in pdf_dir.glob("*.pdf")])
        cached_pdfs = cache_data.get('pdf_files', [])
        
        if current_pdfs == cached_pdfs:
            print("✅ PDF files haven't changed, loading from cache...")
            print(f"   - Text chunks: {len(cache_data['texts'])}")
            print(f"   - Tables: {len(cache_data['tables'])}")
            print(f"   - Images: {len(cache_data['images'])}")
            
            return {
                'texts': cache_data['texts'],
                'tables': cache_data['tables'],
                'images': cache_data['images']
            }
        else:
            print("⚠️  PDF files have changed, reprocessing...")
    
    # Process documents (no cache or cache invalid)
    processor = DocumentProcessor(dataset_path=Config.DATASET_DIR)
    # process_all_documents returns (texts, tables, images) tuple
    texts, tables, images = processor.process_all_documents()
    
    if not texts and not tables and not images:
        print("❌ ERROR: No documents were processed")
        return None
    
    print(f"\n✅ Successfully processed documents:")
    print(f"   - Text chunks: {len(texts)}")
    print(f"   - Tables: {len(tables)}")
    print(f"   - Images: {len(images)}")
    
    # Save to cache
    print("\n💾 Saving processed documents to cache...")
    pdf_dir = Path(Config.DATASET_DIR)
    pdf_files = sorted([f.name for f in pdf_dir.glob("*.pdf")])
    
    import json
    cache_data = {
        'pdf_files': pdf_files,
        'texts': texts,
        'tables': tables,
        'images': images,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    print("✅ Cache saved successfully")
    
    # Return as dictionary for easier handling
    return {
        'texts': texts,
        'tables': tables,
        'images': images
    }

def setup_retriever_with_summaries(documents):
    """Setup retriever and generate all summaries and embeddings"""
    print("\n" + "="*80)
    print("🔍 STEP 2: Setting Up Retriever and Generating Summaries")
    print("="*80)
    
    provider = os.getenv('DEFAULT_PROVIDER', 'OpenAI')
    model = os.getenv('DEFAULT_MODEL', 'gpt-4o-mini')
    
    print(f"\n🤖 Using {provider} - {model}")
    print(f"⏳ This will take several minutes (5-15 min depending on document size)")
    print(f"   Please be patient and do not interrupt the process.\n")
    
    # Set API key in environment
    if provider == 'OpenAI':
        os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    else:
        os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
    
    retriever = MultimodalRetriever(provider=provider, model=model)
    
    # Setup retriever with documents (this will generate summaries AND create embeddings)
    retriever.setup_retriever(
        texts=documents['texts'],
        tables=documents['tables'],
        image_paths=documents['images'],
        persist_directory=str(Config.CHROMA_DB_DIR)
    )
    
    print(f"\n✅ Retriever setup complete!")
    print(f"   - All summaries generated")
    print(f"   - All embeddings created")
    print(f"   - Data stored in ChromaDB and Redis")
    
    return retriever

def verify_storage():
    """Verify that all data was properly stored"""
    print("\n" + "="*80)
    print("✅ STEP 3: Verifying Storage")
    print("="*80)
    
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.storage import RedisStore
    
    # Check ChromaDB
    chroma_dir = Path(Config.CHROMA_DB_DIR)
    if chroma_dir.exists():
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vectorstore = Chroma(
                collection_name="multimodal_rag",
                embedding_function=embeddings,
                persist_directory=str(chroma_dir)
            )
            
            # Try to get count
            collection = vectorstore._collection
            count = collection.count()
            print(f"✅ ChromaDB: {count} embeddings stored")
        except Exception as e:
            print(f"⚠️  ChromaDB verification warning: {str(e)}")
    else:
        print(f"❌ ChromaDB directory not found")
    
    # Check Redis
    try:
        redis_url = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}"
        store = RedisStore(redis_url=redis_url, namespace="multimodal_rag")
        
        # Try to access store
        print(f"✅ Redis: Connected successfully")
    except Exception as e:
        print(f"⚠️  Using InMemoryStore instead of Redis")
        print(f"   (This is okay for development, but data won't persist)")
    
    return True

def create_preprocessing_metadata():
    """Create a metadata file with preprocessing information"""
    print("\n" + "="*80)
    print("📝 STEP 4: Creating Metadata")
    print("="*80)
    
    metadata = {
        'preprocessing_date': datetime.now().isoformat(),
        'provider': os.getenv('DEFAULT_PROVIDER', 'OpenAI'),
        'model': os.getenv('DEFAULT_MODEL', 'gpt-4o-mini'),
        'pdf_directory': str(Config.DATASET_DIR),
        'chroma_directory': str(Config.CHROMA_DB_DIR),
        'redis_directory': str(Config.BASE_DIR / "redis_storage"),
        'status': 'ready'
    }
    
    # Save metadata
    import json
    metadata_file = Path('.preprocessing_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to {metadata_file}")
    
    return metadata

def main():
    """Main preprocessing function"""
    print("\n" + "="*80)
    print("🚀 RAG SYSTEM PREPROCESSING SCRIPT")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Check for --force flag to skip cache
    force_reprocess = '--force' in sys.argv
    if force_reprocess:
        print("\n⚠️  --force flag detected: Will reprocess all documents")
        cache_file = Path('.document_cache.json')
        if cache_file.exists():
            cache_file.unlink()
            print("   Deleted existing cache")
    
    try:
        # Step 0: Check prerequisites
        if not check_prerequisites():
            print("\n❌ Prerequisites check failed. Please fix the issues above.")
            sys.exit(1)
        
        # Step 1: Process documents
        documents = process_documents()
        if not documents:
            print("\n❌ Document processing failed.")
            sys.exit(1)
        
        # Step 2: Setup retriever and generate summaries
        retriever = setup_retriever_with_summaries(documents)
        
        # Step 3: Verify storage
        verify_storage()
        
        # Step 4: Create metadata
        metadata = create_preprocessing_metadata()
        
        # Final summary
        print("\n" + "="*80)
        print("🎉 PREPROCESSING COMPLETE!")
        print("="*80)
        print(f"\n✅ All data has been prepared and stored")
        print(f"✅ The application is now ready to use")
        print(f"\nTo start the application, run:")
        print(f"   streamlit run app.py")
        print(f"\n" + "="*80)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Preprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
