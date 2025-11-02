"""
Test script to check retriever output
"""
import os
from dotenv import load_dotenv
from retriever import MultimodalRetriever

load_dotenv()

# Set API key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Create retriever
retriever = MultimodalRetriever(provider='OpenAI', model='gpt-4o-mini')

# Load from storage
try:
    retriever.load_from_storage('chroma_db')
    
    # Test retrieval
    print("\n" + "="*80)
    print("Testing retrieval...")
    print("="*80)
    
    results = retriever.retrieve("What is a Transformer?", k=2)
    
    print(f"\nRetrieved {len(results)} results")
    print("\nFirst result structure:")
    if results:
        first = results[0]
        print(f"Type: {type(first)}")
        print(f"Keys: {first.keys() if isinstance(first, dict) else 'Not a dict'}")
        if isinstance(first, dict):
            print(f"\nContent preview: {str(first.get('content', 'N/A'))[:200]}...")
            print(f"\nSummary preview: {str(first.get('summary', 'N/A'))[:200]}...")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
