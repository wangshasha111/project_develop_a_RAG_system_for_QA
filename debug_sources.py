"""
Debug script to check what's in the sources
"""
import os
from dotenv import load_dotenv
from retriever import MultimodalRetriever
from rag_chain import RAGChain

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Create retriever
retriever = MultimodalRetriever(provider='OpenAI', model='gpt-4o-mini')
retriever.load_from_storage('chroma_db')

# Create RAG chain
rag_chain = RAGChain(retriever=retriever, provider='OpenAI', model='gpt-4o-mini')

# Query
response = rag_chain.query("What is a Transformer?", k=2)

print("\n" + "="*80)
print("DEBUGGING SOURCES STRUCTURE")
print("="*80)

sources = response['sources']

print(f"\n1. Sources type: {type(sources)}")
print(f"2. Sources keys: {sources.keys()}")

if 'texts' in sources:
    texts = sources['texts']
    print(f"\n3. Texts type: {type(texts)}")
    print(f"4. Number of texts: {len(texts)}")
    
    if texts:
        first = texts[0]
        print(f"\n5. First text type: {type(first)}")
        
        if isinstance(first, dict):
            print(f"6. First text keys: {first.keys()}")
            print(f"\n7. First text content:")
            print(f"   - Type: {type(first.get('content'))}")
            print(f"   - Length: {len(str(first.get('content', '')))}")
            print(f"   - Preview: {str(first.get('content', ''))[:100]}...")
            
            print(f"\n8. First text summary:")
            print(f"   - Type: {type(first.get('summary'))}")
            print(f"   - Length: {len(str(first.get('summary', '')))}")
            print(f"   - Preview: {str(first.get('summary', ''))[:100]}...")
            print(f"   - Is empty: {not first.get('summary')}")
            print(f"   - Bool value: {bool(first.get('summary'))}")
        else:
            print(f"6. First text value: {str(first)[:100]}...")

print("\n" + "="*80)
print("This is what app.py should receive in message['sources']")
print("="*80)
