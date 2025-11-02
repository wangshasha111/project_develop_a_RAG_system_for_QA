"""
RAG Chain Module
Handles the end-to-end RAG pipeline with multimodal support
"""

import re
import base64
from typing import List, Dict, Any, Optional
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


class RAGChain:
    """Multimodal RAG chain for question answering"""
    
    def __init__(self, retriever, provider: str = "OpenAI", model: str = "gpt-4o"):
        """
        Initialize RAG chain
        
        Args:
            retriever: MultimodalRetriever instance
            provider: AI provider ("OpenAI" or "Google")
            model: Model name to use
        """
        self.retriever = retriever
        self.provider = provider
        self.model = model
        
        # Initialize chat model
        if provider == "OpenAI":
            from langchain_openai import ChatOpenAI
            self.chat_model = ChatOpenAI(model_name=model, temperature=0)
        else:  # Google
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.chat_model = ChatGoogleGenerativeAI(model=model, temperature=0)
    
    @staticmethod
    def looks_like_base64(sb: str) -> bool:
        """Check if string looks like base64"""
        return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None
    
    @staticmethod
    def is_image_data(b64data: str) -> bool:
        """
        Check if base64 data is an image by examining the header
        
        Args:
            b64data: Base64 encoded string
            
        Returns:
            True if data is an image
        """
        image_signatures = {
            b"\xff\xd8\xff": "jpg",
            b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
            b"\x47\x49\x46\x38": "gif",
            b"\x52\x49\x46\x46": "webp",
        }
        try:
            header = base64.b64decode(b64data)[:8]
            for sig, _ in image_signatures.items():
                if header.startswith(sig):
                    return True
            return False
        except Exception:
            return False
    
    def split_image_text_types(self, docs: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Split retrieved documents into images and text, preserving summaries
        
        Args:
            docs: List of retrieved documents (dictionaries with 'content' and 'summary')
            
        Returns:
            Dictionary with 'images', 'texts', and 'tables' keys, each containing list of dicts
        """
        images = []
        texts = []
        tables = []
        
        for doc in docs:
            # Handle both new format (dict) and old format (bytes) for backward compatibility
            if isinstance(doc, dict):
                content = doc.get('content', b'')
                summary = doc.get('summary', '')
            else:
                content = doc
                summary = ''
            
            # Decode bytes to string
            if isinstance(content, bytes):
                try:
                    content_str = content.decode('utf-8')
                except UnicodeDecodeError:
                    # If can't decode, might be binary image data
                    content_str = base64.b64encode(content).decode('utf-8')
            else:
                content_str = str(content)
            
            # Check if it's an image
            if self.looks_like_base64(content_str) and self.is_image_data(content_str):
                images.append({
                    'content': content_str,
                    'summary': summary
                })
            # Check if it's a table (contains markdown table syntax or HTML table)
            elif '|' in content_str and ('---' in content_str or '|-' in content_str):
                tables.append({
                    'content': content_str,
                    'summary': summary
                })
            else:
                texts.append({
                    'content': content_str,
                    'summary': summary
                })
        
        return {"images": images, "texts": texts, "tables": tables}
    
    def create_multimodal_prompt(
        self, 
        context: Dict[str, List[Dict]], 
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> List[HumanMessage]:
        """
        Create multimodal prompt with text and images
        
        Args:
            context: Dictionary with 'texts', 'tables', and 'images' (each a list of dicts with 'content' and 'summary')
            question: User question
            chat_history: Optional chat history
            
        Returns:
            List of HumanMessage for the LLM
        """
        messages = []
        
        # Add images to messages
        if context.get("images"):
            for img_dict in context["images"]:
                image_content = img_dict.get('content', img_dict) if isinstance(img_dict, dict) else img_dict
                image_message = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_content}"},
                }
                messages.append(image_message)
        
        # Format chat history if provided
        history_text = ""
        if chat_history:
            history_text = "\n\nPrevious conversation:\n"
            for msg in chat_history[-3:]:  # Include last 3 exchanges
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                history_text += f"{role.upper()}: {content}\n"
        
        # Format context texts (extract content from dicts)
        text_contents = []
        for text_dict in context.get("texts", []):
            if isinstance(text_dict, dict):
                text_contents.append(text_dict.get('content', ''))
            else:
                text_contents.append(str(text_dict))
        formatted_texts = "\n\n".join(text_contents)
        
        # Format tables
        table_contents = []
        for table_dict in context.get("tables", []):
            if isinstance(table_dict, dict):
                table_contents.append(table_dict.get('content', ''))
            else:
                table_contents.append(str(table_dict))
        formatted_tables = "\n\n".join(table_contents)
        
        # Add tables to context if available
        table_section = ""
        if formatted_tables:
            table_section = f"\n\nTABLES:\n{formatted_tables}"
        
        # Create main text message
        text_message = {
            "type": "text",
            "text": f"""You are an expert AI assistant that ONLY answers questions based on the three provided research papers about Transformers and attention mechanisms.

⚠️ CRITICAL RESTRICTIONS:
1. You can ONLY use information from the provided context (text, tables, images)
2. If the answer is NOT found in the provided context, you MUST respond: "I don't know. This information is not covered in the available documents."
3. Do NOT use external knowledge, assumptions, or information from other sources
4. Do NOT make up or infer information that is not explicitly stated in the context

INSTRUCTIONS FOR ANSWERING:
1. Analyze ALL provided context including text, tables, and images
2. Provide comprehensive answers ONLY from the sources
3. Always cite the source (e.g., "According to the Attention Is All You Need paper...", "The table shows...")
4. If the context contains tables or numerical data, include those details
5. If images show charts or graphs, describe the trends shown
6. If you cannot find a clear answer in the context, say "I don't know" or "This is not covered in the provided documents"
7. Do NOT make up information that is not in the context

{history_text}

CURRENT QUESTION:
{question}

CONTEXT DOCUMENTS:
{formatted_texts}{table_section}

Provide a detailed, well-structured answer:"""
        }
        messages.append(text_message)
        
        return [HumanMessage(content=messages)]
    
    def query(
        self, 
        question: str, 
        k: int = 5,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            k: Number of documents to retrieve
            chat_history: Optional chat history
            
        Returns:
            Dictionary with 'answer' and 'sources'
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question, k=k)
        
        # Split into images and texts
        context = self.split_image_text_types(retrieved_docs)
        
        # Create prompt
        prompt_messages = self.create_multimodal_prompt(context, question, chat_history)
        
        # Generate answer
        response = self.chat_model.invoke(prompt_messages)
        
        # Extract answer
        if hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)
        
        return {
            'answer': answer,
            'sources': context
        }
    
    def build_chain(self):
        """
        Build the full RAG chain as a LangChain Runnable
        
        Returns:
            RAG chain runnable
        """
        # Retrieve documents
        def retrieve_docs(inputs):
            query = inputs.get('input', inputs.get('question', ''))
            docs = self.retriever.retrieve(query)
            return self.split_image_text_types(docs)
        
        # Create prompt function
        def create_prompt(inputs):
            return self.create_multimodal_prompt(
                inputs['context'],
                inputs['question'],
                inputs.get('chat_history')
            )
        
        # Build chain
        retrieve = RunnableLambda(retrieve_docs)
        prompt_func = RunnableLambda(create_prompt)
        
        # Chain: retrieve -> create prompt -> LLM -> parse output
        chain = (
            RunnablePassthrough.assign(context=retrieve)
            .assign(
                prompt=lambda x: create_prompt({
                    'context': x['context'],
                    'question': x.get('input', x.get('question', '')),
                    'chat_history': x.get('chat_history')
                })
            )
            | {
                'answer': lambda x: self.chat_model.invoke(x['prompt']),
                'sources': lambda x: x['context']
            }
        )
        
        return chain


def get_mock_rag_chain():
    """
    Create a mock RAG chain for testing
    
    Returns:
        Mock RAG chain object
    """
    class MockRAGChain:
        def query(self, question: str, k: int = 5, chat_history: Optional[List] = None):
            # Generate mock response based on question
            if "transformer" in question.lower() or "encoder" in question.lower():
                answer = """The Transformer model consists of an encoder and decoder architecture. 
Each encoder layer has two sub-layers:

1. **Multi-head self-attention mechanism**: This allows the model to attend to different positions 
   of the input sequence simultaneously. The multi-head attention runs several attention mechanisms 
   in parallel, allowing the model to focus on different representation subspaces.

2. **Position-wise fully connected feed-forward network**: This is applied to each position 
   separately and identically. It consists of two linear transformations with a ReLU activation 
   in between.

Each sub-layer has a residual connection around it, followed by layer normalization. This architecture 
is repeated N times (typically 6 layers) in both the encoder and decoder."""
                
            elif "rag" in question.lower():
                answer = """RAG (Retrieval-Augmented Generation) models combine the benefits of retrieval-based 
and generation-based approaches. The main components are:

1. **Retriever**: Searches through a large corpus of documents to find relevant information based on 
   the input query. This typically uses dense vector representations and similarity search.

2. **Generator**: A language model (like GPT or T5) that generates the final answer based on both 
   the query and the retrieved documents.

3. **Document Store**: A database or index containing the knowledge base that the retriever searches through.

These components interact as follows:
- The user query is encoded and used to retrieve relevant documents
- Retrieved documents are provided as context to the generator
- The generator produces an answer that combines information from multiple sources
- This allows the model to access external knowledge while maintaining the fluency of generation"""
                
            elif "positional" in question.lower():
                answer = """Positional encoding is implemented in Transformers using sinusoidal functions:

**Implementation**:
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
- pos is the position in the sequence
- i is the dimension
- d_model is the model dimension

**Why it's necessary**:
1. The self-attention mechanism is permutation-invariant - it treats the input as a set, not a sequence
2. Without positional information, the model cannot distinguish between different orderings
3. Positional encodings inject information about the relative or absolute position of tokens
4. The sinusoidal functions allow the model to easily learn to attend to relative positions"""
                
            elif "multi-head" in question.lower():
                answer = """Multi-head attention is a key component of the Transformer architecture:

**Concept**:
Instead of performing a single attention function, multi-head attention runs multiple attention 
mechanisms in parallel (typically 8 heads). Each head can focus on different aspects of the input.

**Benefits**:
1. **Diverse Representations**: Different heads can learn to attend to different positions and 
   capture different types of relationships
2. **Richer Context**: By attending to multiple subspaces, the model can capture more complex patterns
3. **Parallel Processing**: Multiple heads can be computed in parallel, improving efficiency
4. **Better Performance**: Empirically shown to improve performance across various NLP tasks

The outputs from all heads are concatenated and linearly transformed to produce the final output."""
                
            elif "few-shot" in question.lower() or "gpt-3" in question.lower():
                answer = """Few-shot learning in GPT-3:

**What is few-shot learning**:
Few-shot learning is the ability to perform a task with only a few examples, without explicit training 
or fine-tuning on that specific task.

**GPT-3 Implementation**:
1. **In-Context Learning**: GPT-3 uses examples provided in the prompt itself as demonstrations
2. **Prompt Format**: The model is given a few input-output pairs as examples, followed by a new input
3. **Pattern Recognition**: GPT-3 recognizes the pattern from the examples and applies it to generate 
   the output for the new input

**Example**:
```
Translate to French:
English: Hello -> French: Bonjour
English: Goodbye -> French: Au revoir
English: Thank you -> French: [GPT-3 generates] Merci
```

This emergent ability comes from GPT-3's massive scale (175B parameters) and diverse training data."""
                
            else:
                answer = """Based on the research papers, I can provide information about transformer 
architectures, RAG systems, attention mechanisms, and language models. Please ask a specific question 
about these topics for a detailed answer."""
            
            # Mock sources
            sources = {
                'texts': [
                    "Source 1: Attention Is All You Need - Transformer Architecture",
                    "Source 2: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                    "Source 3: Language Models are Few-Shot Learners (GPT-3 Paper)"
                ],
                'images': []  # No images in mock mode
            }
            
            return {
                'answer': answer,
                'sources': sources
            }
    
    return MockRAGChain()
