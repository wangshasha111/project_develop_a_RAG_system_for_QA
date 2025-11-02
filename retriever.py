"""
Multi-Vector Retriever Module
Handles document embedding, storage, and retrieval using Chroma and Redis
"""

import os
import uuid
import base64
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.storage import RedisStore
from langchain_community.utilities.redis import get_client
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage


class MultimodalRetriever:
    """Multimodal retriever for text, tables, and images"""
    
    def __init__(self, provider: str = "OpenAI", model: str = "gpt-4o"):
        """
        Initialize the multimodal retriever
        
        Args:
            provider: AI provider ("OpenAI" or "Google")
            model: Model name to use
        """
        self.provider = provider
        self.model = model
        self.retriever = None
        self.vectorstore = None
        self.docstore = None
        
        # Initialize embeddings and chat model based on provider
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding and chat models based on provider"""
        if self.provider == "OpenAI":
            from langchain_openai import OpenAIEmbeddings, ChatOpenAI
            self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
            self.chat_model = ChatOpenAI(model_name=self.model, temperature=0)
        else:  # Google
            from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            self.chat_model = ChatGoogleGenerativeAI(model=self.model, temperature=0)
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _summarize_text(self, text: str) -> str:
        """
        Generate summary for text using LLM
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary string
        """
        prompt_text = """
You are an assistant tasked with summarizing tables and text particularly for semantic retrieval.
These summaries will be embedded and used to retrieve the raw text or table elements.
Give a detailed summary of the table or text below that is well optimized for retrieval.
For any tables also add in a one line description of what the table is about besides the summary.
Do not add redundant words like Summary.
Just output the actual summary content.

Table or text chunk:
{element}
"""
        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = (
            {"element": RunnablePassthrough()}
            | prompt
            | self.chat_model
            | StrOutputParser()
        )
        
        return chain.invoke(text)
    
    def _summarize_image(self, img_base64: str) -> str:
        """
        Generate summary for image using multimodal LLM
        
        Args:
            img_base64: Base64 encoded image
            
        Returns:
            Image summary string
        """
        prompt = """You are an assistant tasked with summarizing images for retrieval.
Remember these images could potentially contain graphs, charts or tables also.
These summaries will be embedded and used to retrieve the raw image for question answering.
Give a detailed summary of the image that is well optimized for retrieval.
Do not add additional words like Summary, This image represents, etc."""

        if self.provider == "OpenAI":
            msg = self.chat_model.invoke([
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )
            ])
            return msg.content
        else:  # Google Gemini
            # For Gemini, we need to use a different approach
            from langchain_google_genai import ChatGoogleGenerativeAI
            import google.generativeai as genai
            
            # Initialize Gemini with vision capabilities
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Decode base64 to bytes
            img_bytes = base64.b64decode(img_base64)
            
            # Create image part
            from PIL import Image
            from io import BytesIO
            img = Image.open(BytesIO(img_bytes))
            
            # Generate summary
            response = model.generate_content([prompt, img])
            return response.text
    
    def generate_summaries(
        self, 
        texts: List[str], 
        tables: List[str], 
        image_paths: List[str]
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Generate summaries for all document elements
        
        Args:
            texts: List of text chunks
            tables: List of table content (markdown)
            image_paths: List of image file paths
            
        Returns:
            Tuple of (text_summaries, table_summaries, image_summaries, images_base64)
        """
        print("\n" + "="*60)
        print("🧠 GENERATING SUMMARIES FOR RETRIEVAL")
        print("="*60)
        
        # Generate text summaries
        text_summaries = []
        if texts:
            print(f"\n📝 Summarizing {len(texts)} text chunks...")
            print("⏳ This may take several minutes depending on chunk count...")
            for i, text in enumerate(texts):
                try:
                    summary = self._summarize_text(text)
                    text_summaries.append(summary)
                    if (i + 1) % 5 == 0:
                        print(f"  ✓ Processed {i + 1}/{len(texts)} text chunks ({((i+1)/len(texts)*100):.1f}%)")
                except Exception as e:
                    print(f"    ⚠️  Warning: Error summarizing text {i}: {e}")
                    text_summaries.append(text[:500])  # Use truncated text as fallback
            print(f"✅ Completed {len(text_summaries)} text summaries")
        
        # Generate table summaries
        table_summaries = []
        if tables:
            print(f"\n📊 Summarizing {len(tables)} tables...")
            for i, table in enumerate(tables):
                try:
                    summary = self._summarize_text(table)
                    table_summaries.append(summary)
                    print(f"  ✓ Processed table {i + 1}/{len(tables)}")
                except Exception as e:
                    print(f"    ⚠️  Warning: Error summarizing table {i}: {e}")
                    table_summaries.append(table[:500])
            print(f"✅ Completed {len(table_summaries)} table summaries")
        
        # Encode images and generate summaries
        images_base64 = []
        image_summaries = []
        if image_paths:
            print(f"\n🖼️  Processing {len(image_paths)} images...")
            print("⏳ Encoding and summarizing images...")
            for i, img_path in enumerate(image_paths):
                try:
                    # Encode image
                    img_b64 = self._encode_image(img_path)
                    images_base64.append(img_b64)
                    
                    # Generate summary
                    summary = self._summarize_image(img_b64)
                    image_summaries.append(summary)
                    
                    print(f"  ✓ Processed image {i + 1}/{len(image_paths)} ({((i+1)/len(image_paths)*100):.1f}%)")
                except Exception as e:
                    print(f"    ⚠️  Warning: Error processing image {i}: {e}")
            print(f"✅ Completed {len(image_summaries)} image summaries")
        
        print("\n" + "="*60)
        print(f"✅ SUMMARY GENERATION COMPLETE")
        print("="*60)
        print(f"📊 Results:")
        print(f"  • Text summaries: {len(text_summaries)}")
        print(f"  • Table summaries: {len(table_summaries)}")
        print(f"  • Image summaries: {len(image_summaries)}")
        print("="*60 + "\n")
        
        return text_summaries, table_summaries, image_summaries, images_base64
    
    def setup_retriever(
        self,
        texts: List[str],
        tables: List[str],
        image_paths: List[str],
        persist_directory: Optional[str] = None
    ):
        """
        Setup the multi-vector retriever with documents
        
        Args:
            texts: List of text chunks
            tables: List of table content
            image_paths: List of image file paths
            persist_directory: Optional directory to persist vectorstore
        """
        # Generate summaries
        text_summaries, table_summaries, image_summaries, images_base64 = \
            self.generate_summaries(texts, tables, image_paths)
        
        # Initialize vectorstore
        if persist_directory:
            persist_path = Path(persist_directory)
            persist_path.mkdir(exist_ok=True)
        else:
            persist_path = Path(__file__).parent / "chroma_db"
            persist_path.mkdir(exist_ok=True)
        
        self.vectorstore = Chroma(
            collection_name="mm_rag",
            embedding_function=self.embeddings,
            persist_directory=str(persist_path),
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize docstore (Redis or in-memory fallback)
        use_redis = False
        try:
            client = get_client('redis://localhost:6379')
            # Test connection
            client.ping()
            self.docstore = RedisStore(client=client)
            use_redis = True
            print("✅ Using Redis as document store")
        except Exception as e:
            print(f"⚠️  Could not connect to Redis: {e}")
            print("✅ Using InMemoryStore instead (data won't persist across sessions)")
            from langchain.storage import InMemoryStore
            self.docstore = InMemoryStore()
        
        # Create multi-vector retriever
        id_key = "doc_id"
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=id_key,
        )
        
        # Helper function to add documents with error handling
        # Also save doc_id to content mapping for later loading
        doc_id_mapping = {}
        
        def add_documents(doc_summaries: List[str], doc_contents: List[str]):
            if not doc_summaries or not doc_contents:
                return
            
            doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
            summary_docs = [
                Document(page_content=s, metadata={id_key: doc_ids[i]})
                for i, s in enumerate(doc_summaries)
            ]
            self.retriever.vectorstore.add_documents(summary_docs)
            
            # Encode contents to bytes for storage
            encoded_contents = [c.encode('utf-8') if isinstance(c, str) else c 
                              for c in doc_contents]
            
            # Save mapping for later loading
            for doc_id, content in zip(doc_ids, encoded_contents):
                # Store as base64 string for JSON serialization
                if isinstance(content, bytes):
                    doc_id_mapping[doc_id] = base64.b64encode(content).decode('utf-8')
                else:
                    doc_id_mapping[doc_id] = content
            
            # Add to docstore with error handling
            try:
                self.retriever.docstore.mset(list(zip(doc_ids, encoded_contents)))
            except Exception as e:
                print(f"    ⚠️  Error storing documents: {e}")
                # If Redis fails, fall back to InMemoryStore
                if use_redis:
                    print("    ⚠️  Falling back to InMemoryStore for this batch")
                    from langchain.storage import InMemoryStore
                    self.docstore = InMemoryStore()
                    self.retriever.docstore = self.docstore
                    self.retriever.docstore.mset(list(zip(doc_ids, encoded_contents)))
        
        # Add text, table, and image summaries with their raw content
        print("Adding documents to retriever...")
        if text_summaries:
            print(f"  Adding {len(text_summaries)} text documents...")
            add_documents(text_summaries, texts)
        
        if table_summaries:
            print(f"  Adding {len(table_summaries)} table documents...")
            add_documents(table_summaries, tables)
        
        if image_summaries:
            print(f"  Adding {len(image_summaries)} image documents...")
            add_documents(image_summaries, images_base64)
        
        # Save doc_id mapping to file for loading without Redis
        if doc_id_mapping:
            import json
            mapping_file = persist_path / "doc_id_mapping.json"
            print(f"\n💾 Saving document ID mapping to {mapping_file}...")
            with open(mapping_file, 'w') as f:
                json.dump(doc_id_mapping, f)
            print(f"   ✅ Saved {len(doc_id_mapping)} document mappings")
        
        print("Retriever setup complete!")
    
    def load_from_storage(self, persist_directory: str):
        """
        Load retriever from existing ChromaDB and Redis storage
        
        Args:
            persist_directory: Directory where ChromaDB is persisted
        """
        print(f"\n📂 Loading retriever from existing storage...")
        print(f"   ChromaDB: {persist_directory}")
        
        persist_path = Path(persist_directory)
        if not persist_path.exists():
            raise ValueError(f"Storage directory not found: {persist_directory}")
        
        # Load vectorstore
        self.vectorstore = Chroma(
            collection_name="mm_rag",
            embedding_function=self.embeddings,
            persist_directory=str(persist_path),
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # Try to load docstore (Redis first, then fallback to mapping file)
        redis_connected = False
        try:
            client = get_client('redis://localhost:6379')
            client.ping()
            self.docstore = RedisStore(client=client)
            redis_connected = True
            print("   ✅ Connected to Redis docstore")
        except Exception as e:
            print(f"   ⚠️  Could not connect to Redis: {e}")
            
            # Load from mapping file
            mapping_file = persist_path / "doc_id_mapping.json"
            if mapping_file.exists():
                print(f"   📂 Loading document mapping from {mapping_file}...")
                import json
                with open(mapping_file, 'r') as f:
                    doc_id_mapping = json.load(f)
                
                # Create InMemoryStore and populate it
                from langchain.storage import InMemoryStore
                self.docstore = InMemoryStore()
                
                # Decode and store documents
                doc_items = []
                for doc_id, content_b64 in doc_id_mapping.items():
                    # Decode from base64
                    content_bytes = base64.b64decode(content_b64)
                    doc_items.append((doc_id, content_bytes))
                
                if doc_items:
                    self.docstore.mset(doc_items)
                    print(f"   ✅ Loaded {len(doc_items)} documents from mapping file")
            else:
                print(f"   ❌ No mapping file found at {mapping_file}")
                print(f"   ⚠️  Using empty InMemoryStore (retrieval may fail)")
                from langchain.storage import InMemoryStore
                self.docstore = InMemoryStore()
        
        # Create multi-vector retriever
        id_key = "doc_id"
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=id_key,
        )
        
        # Check how many documents are loaded
        try:
            count = self.vectorstore._collection.count()
            print(f"   ✅ Loaded {count} document summaries from storage")
        except Exception as e:
            print(f"   ⚠️  Could not get document count: {e}")
        
        print("✅ Retriever loaded successfully!\n")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of dictionaries with 'content' (original) and 'summary' keys
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Call setup_retriever or load_from_storage first.")
        
        # First, search the vectorstore to get relevant summaries
        summary_docs = self.vectorstore.similarity_search(query, k=k)
        
        # Get the original documents using doc_ids
        results = []
        for summary_doc in summary_docs:
            doc_id = summary_doc.metadata.get('doc_id')
            summary = summary_doc.page_content
            
            # Retrieve original content from docstore
            if doc_id and self.docstore:
                try:
                    original_content = self.docstore.mget([doc_id])[0]
                    if original_content:
                        results.append({
                            'content': original_content,
                            'summary': summary
                        })
                    else:
                        # If no original content, use summary as content
                        results.append({
                            'content': summary.encode('utf-8'),
                            'summary': summary
                        })
                except Exception as e:
                    # Fallback: use summary as content if retrieval fails
                    results.append({
                        'content': summary.encode('utf-8'),
                        'summary': summary
                    })
            else:
                # Fallback if no doc_id or docstore
                results.append({
                    'content': summary.encode('utf-8'),
                    'summary': summary
                })
        
        return results


def get_mock_retriever():
    """
    Create a mock retriever for testing
    
    Returns:
        Mock retriever object
    """
    class MockRetriever:
        def retrieve(self, query: str, k: int = 5):
            # Return mock documents
            mock_docs = [
                b"The Transformer architecture uses self-attention mechanisms...",
                b"Multi-head attention allows the model to focus on different positions...",
                b"RAG systems combine retrieval with generation for better accuracy..."
            ]
            return mock_docs[:k]
    
    return MockRetriever()
