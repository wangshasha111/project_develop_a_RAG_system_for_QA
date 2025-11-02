"""
Test the full RAG chain to see data structure
"""
import os
from dotenv import load_dotenv
from retriever import MultimodalRetriever
from rag_chain import RAGChain

load_dotenv()

# Set API key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Create retriever
retriever = MultimodalRetriever(provider='OpenAI', model='gpt-4o-mini')
retriever.load_from_storage('chroma_db')

# Create RAG chain
rag_chain = RAGChain(retriever=retriever, provider='OpenAI', model='gpt-4o-mini')

# Query
print("\n" + "="*80)
print("Testing RAG chain...")
print("="*80)

response = rag_chain.query("What is a Transformer?", k=2)

print(f"\nResponse keys: {response.keys()}")
print(f"\nAnswer preview: {response['answer'][:200]}...")
print(f"\nSources keys: {response['sources'].keys()}")
print(f"\nNumber of texts: {len(response['sources']['texts'])}")
print(f"Number of tables: {len(response['sources']['tables'])}")
print(f"Number of images: {len(response['sources']['images'])}")

if response['sources']['texts']:
    first_text = response['sources']['texts'][0]
    print(f"\nFirst text type: {type(first_text)}")
    if isinstance(first_text, dict):
        print(f"First text keys: {first_text.keys()}")
        print(f"Has summary: {'summary' in first_text and bool(first_text['summary'])}")
        print(f"Summary length: {len(first_text.get('summary', ''))}")
        print(f"Content length: {len(str(first_text.get('content', '')))}")
        print(f"\nSummary preview: {first_text.get('summary', 'N/A')[:200]}...")
    else:
        print(f"First text value: {str(first_text)[:200]}...")
