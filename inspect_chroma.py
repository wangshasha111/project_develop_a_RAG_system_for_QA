"""
Check what's in ChromaDB and see if we can extract the data
"""
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

print("\n" + "="*80)
print("🔍 INSPECTING CHROMADB")
print("="*80)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="mm_rag",
    embedding_function=embeddings,
    persist_directory="chroma_db",
    collection_metadata={"hnsw:space": "cosine"}
)

collection = vectorstore._collection
all_docs = collection.get(include=['metadatas', 'documents'])

print(f"\n📊 Total documents in ChromaDB: {len(all_docs['ids'])}")

if all_docs['ids']:
    print(f"\n📋 First 3 documents:")
    for i in range(min(3, len(all_docs['ids']))):
        print(f"\n--- Document {i+1} ---")
        print(f"ID: {all_docs['ids'][i]}")
        print(f"Metadata: {all_docs['metadatas'][i]}")
        print(f"Summary (first 200 chars): {all_docs['documents'][i][:200]}...")

print("\n" + "="*80)
