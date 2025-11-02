"""
Regenerate doc_id_mapping.json from existing ChromaDB and document cache
This script extracts summaries from ChromaDB and original content from cache
"""

import os
import json
import base64
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Set API key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

print("\n" + "="*80)
print("🔄 REGENERATING doc_id_mapping.json")
print("="*80)

# Check if we have the document cache with original content
cache_file = Path('.document_cache.json')
if not cache_file.exists():
    print("\n❌ ERROR: .document_cache.json not found")
    print("   Please run: python preprocess.py")
    exit(1)

print("\n📂 Loading cached documents...")
with open(cache_file, 'r') as f:
    cache_data = json.load(f)

texts = cache_data.get('texts', [])
tables = cache_data.get('tables', [])
images = cache_data.get('images', [])

print(f"   - Text chunks: {len(texts)}")
print(f"   - Tables: {len(tables)}")
print(f"   - Images: {len(images)}")

# Load ChromaDB to get summaries and doc_ids
print("\n📂 Loading ChromaDB...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="mm_rag",
    embedding_function=embeddings,
    persist_directory="chroma_db",
    collection_metadata={"hnsw:space": "cosine"}
)

# Get all documents from ChromaDB
collection = vectorstore._collection
all_docs = collection.get(include=['metadatas', 'documents'])

print(f"   - Found {len(all_docs['ids'])} documents in ChromaDB")

# Create mapping: doc_id -> original content (encoded as base64)
doc_id_mapping = {}

# We need to match summaries in ChromaDB with original content
# Since we don't have the exact mapping, we'll use the order
all_content = texts + tables + images

if len(all_docs['ids']) != len(all_content):
    print(f"\n⚠️  WARNING: Mismatch in document counts:")
    print(f"   ChromaDB: {len(all_docs['ids'])}")
    print(f"   Cache: {len(all_content)}")
    print(f"   Using minimum of both...")

# Create mapping
for i, doc_id in enumerate(all_docs['ids']):
    if i < len(all_content):
        content = all_content[i]
        # Encode content to bytes if it's a string
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        else:
            content_bytes = content
        
        # Store as base64 string
        doc_id_mapping[doc_id] = base64.b64encode(content_bytes).decode('utf-8')

print(f"\n💾 Saving doc_id_mapping.json...")
mapping_file = Path('chroma_db/doc_id_mapping.json')
with open(mapping_file, 'w') as f:
    json.dump(doc_id_mapping, f)

print(f"✅ Saved {len(doc_id_mapping)} mappings to {mapping_file}")

print("\n" + "="*80)
print("🎉 MAPPING REGENERATION COMPLETE!")
print("="*80)
print("\nNow restart the Streamlit app to see summaries and original content!")
