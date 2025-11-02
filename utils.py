"""
Utilities Module
Helper functions for the RAG system
"""

import base64
import re
from typing import Dict, List, Any
from io import BytesIO
from PIL import Image


def encode_image_for_display(image_path: str) -> str:
    """
    Encode image file to base64 for display
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def decode_base64_image(b64_string: str) -> Image:
    """
    Decode base64 string to PIL Image
    
    Args:
        b64_string: Base64 encoded image string
        
    Returns:
        PIL Image object
    """
    img_data = base64.b64decode(b64_string)
    img_buffer = BytesIO(img_data)
    return Image.open(img_buffer)


def looks_like_base64(sb: str) -> bool:
    """
    Check if string looks like base64
    
    Args:
        sb: String to check
        
    Returns:
        True if string appears to be base64 encoded
    """
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


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


def format_chat_history(chat_history: List[Dict[str, str]], max_exchanges: int = 3) -> str:
    """
    Format chat history for inclusion in prompts
    
    Args:
        chat_history: List of chat messages
        max_exchanges: Maximum number of exchanges to include
        
    Returns:
        Formatted chat history string
    """
    if not chat_history:
        return ""
    
    history_text = "\n\nPrevious conversation:\n"
    for msg in chat_history[-max_exchanges * 2:]:  # Get last N exchanges (user + assistant)
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        history_text += f"{role.upper()}: {content}\n"
    
    return history_text


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def get_mock_response(question: str) -> Dict[str, Any]:
    """
    Generate high-quality mock response for testing without API calls.
    Provides comprehensive, well-formatted answers based on actual research papers.
    
    Args:
        question: User question
        
    Returns:
        Dictionary with 'answer' and 'sources' keys
        - answer: Detailed, formatted response with citations
        - sources: Dict with 'texts' and 'images' lists
    """
    question_lower = question.lower()
    
    # Determine response based on question keywords
    if "transformer" in question_lower or "encoder" in question_lower or "layer" in question_lower:
        answer = """**Transformer Encoder Layers**

Based on the research papers, each encoder layer in the Transformer model contains two main sub-layers:

1. **Multi-head Self-Attention Mechanism**
   - This sub-layer allows the model to attend to different positions of the input sequence simultaneously
   - The "multi-head" aspect means it runs several attention mechanisms in parallel (typically 8 heads)
   - Each head can focus on different aspects of the input, capturing various relationships and patterns
   - The outputs from all heads are concatenated and linearly transformed

2. **Position-wise Fully Connected Feed-Forward Network**
   - Applied to each position separately and identically
   - Consists of two linear transformations with a ReLU activation in between
   - Formula: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
   - The same feed-forward network is applied to each position independently

**Additional Components:**
- Each sub-layer has a **residual connection** around it, followed by layer normalization
- The output of each sub-layer is: LayerNorm(x + Sublayer(x))
- This architecture is typically repeated 6 times in the encoder stack

*Source: "Attention Is All You Need" (Vaswani et al., 2017)*"""
        
    elif "rag" in question_lower and "component" in question_lower:
        answer = """**Main Components of a RAG Model**

A Retrieval-Augmented Generation (RAG) model consists of three primary components that work together:

1. **Retriever Component**
   - Searches through a large corpus of documents to find relevant information
   - Uses dense vector representations (embeddings) to encode queries and documents
   - Employs similarity search algorithms (e.g., FAISS, approximate nearest neighbors) to find top-k most relevant documents
   - Can use various retrieval methods: BM25, dense passage retrieval (DPR), or hybrid approaches

2. **Generator Component**
   - A language model (such as BART, T5, or GPT) that generates the final answer
   - Takes as input both the original query and the retrieved documents
   - Produces fluent, coherent text that combines information from multiple sources
   - Can be fine-tuned on specific tasks for better performance

3. **Document Store/Knowledge Base**
   - Contains the corpus of documents or knowledge to search through
   - Can be a vector database (Chroma, Pinecone, Weaviate) or traditional database
   - Stores both original documents and their embeddings
   - May include text, tables, images, and other data types

**How They Interact:**
1. User query → Encoder → Dense vector representation
2. Vector search in document store → Retrieve top-k relevant documents
3. Query + Retrieved docs → Generator → Final answer
4. The system can iterate, using generated output to refine retrieval

This architecture enables the model to access up-to-date external knowledge while maintaining the fluency of neural generation.

*Source: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)*"""
        
    elif "positional" in question_lower:
        answer = """**Positional Encoding in Transformers**

**Implementation:**

Transformers use sinusoidal positional encodings to inject position information:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos` is the position in the sequence (0, 1, 2, ...)
- `i` is the dimension index
- `d_model` is the model dimension (e.g., 512)
- Even dimensions use sine, odd dimensions use cosine

**Why It's Necessary:**

1. **Attention is Permutation-Invariant**
   - The self-attention mechanism treats input as an unordered set
   - Without positional information, "cat sat on mat" would be identical to "mat sat on cat"

2. **No Recurrence or Convolution**
   - Unlike RNNs (which process sequentially) or CNNs (which have positional filters)
   - Transformers process all positions in parallel
   - Position information must be explicitly added

3. **Relative Position Learning**
   - Sinusoidal functions allow the model to easily learn relative positions
   - For any fixed offset k, PE(pos+k) can be represented as a linear function of PE(pos)
   - This helps the model attend to relative positions (e.g., "word 3 positions to the left")

4. **Extrapolation to Longer Sequences**
   - Can generalize to sequence lengths not seen during training
   - Learned positional embeddings would not have this property

The positional encodings are added to the input embeddings at the bottom of the encoder and decoder stacks.

*Source: "Attention Is All You Need" (Vaswani et al., 2017)*"""
        
    elif "multi-head" in question_lower or "attention" in question_lower:
        answer = """**Multi-Head Attention in Transformers**

**Concept:**

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. Instead of performing a single attention function, it runs multiple attention mechanisms in parallel.

**Architecture:**

1. **Parallel Attention Heads**
   - Typically 8 heads (h=8) in the base model
   - Each head has its own learned linear projections for queries (Q), keys (K), and values (V)
   - Dimension per head: d_k = d_v = d_model/h (e.g., 512/8 = 64)

2. **Computation Process**
   ```
   MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O
   where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
   ```

3. **Scaled Dot-Product Attention**
   ```
   Attention(Q,K,V) = softmax(QK^T / √d_k)V
   ```

**Why It's Beneficial:**

1. **Diverse Representations**
   - Different heads can learn to focus on different aspects of the input
   - Some heads might focus on syntax, others on semantics
   - Enables capturing multiple types of relationships simultaneously

2. **Richer Context Understanding**
   - One head might attend to nearby words, another to distant dependencies
   - Captures both local and global patterns
   - Better models complex linguistic phenomena

3. **Improved Performance**
   - Empirically shown to outperform single-head attention
   - More robust and expressive representations
   - Better gradient flow during training

4. **Efficient Parallelization**
   - All heads can be computed in parallel
   - Only a slight increase in computational cost compared to single-head
   - Dimension per head is smaller, keeping total computation similar

**Example Use Cases:**
- In machine translation: some heads focus on source-target alignment, others on syntactic structure
- In question answering: some heads identify question words, others find relevant context

*Source: "Attention Is All You Need" (Vaswani et al., 2017)*"""
        
    elif "few-shot" in question_lower or "gpt-3" in question_lower:
        answer = """**Few-Shot Learning in GPT-3**

**What is Few-Shot Learning?**

Few-shot learning is the ability of a model to perform a new task with only a few examples (typically 1-10), without any gradient updates or fine-tuning. It's a form of meta-learning that demonstrates the model's ability to generalize from minimal data.

**GPT-3's Implementation:**

1. **In-Context Learning**
   - GPT-3 uses examples provided directly in the prompt as demonstrations
   - No parameter updates or fine-tuning required
   - The model learns the task "on the fly" from the context

2. **Prompt Format**
   ```
   Task description (optional)
   
   Example 1: Input → Output
   Example 2: Input → Output
   Example 3: Input → Output
   ...
   
   New Input → [GPT-3 generates output]
   ```

3. **K-Shot Settings**
   - **Zero-shot**: No examples, just task description
   - **One-shot**: 1 example provided
   - **Few-shot**: Typically 10-100 examples
   - Performance generally improves with more examples

**How It Works:**

1. **Pattern Recognition**
   - GPT-3 recognizes patterns in the provided examples
   - Infers the task structure and requirements
   - Applies the learned pattern to generate output for new inputs

2. **Massive Scale Enables Emergence**
   - 175 billion parameters
   - Trained on diverse internet text (570GB of text)
   - Emergent ability - not explicitly trained for few-shot learning
   - Smaller models show much weaker few-shot capabilities

3. **Task Adaptation**
   - Can perform various tasks: translation, QA, arithmetic, word unscrambling
   - Adapts to task format from examples
   - No task-specific architecture changes needed

**Example - Translation:**
```
English: Hello
French: Bonjour

English: How are you?
French: Comment allez-vous?

English: Thank you
French: Merci

English: Good morning
French: [GPT-3 generates] Bon matin
```

**Performance:**
- Approaches or matches fine-tuned models on some tasks
- Particularly strong on tasks requiring broad knowledge
- Struggles with tasks requiring precise reasoning or factual accuracy

**Advantages:**
- No training data collection needed
- Rapid task switching
- Democratizes AI - users don't need ML expertise

**Limitations:**
- Inconsistent performance across tasks
- Sensitive to prompt formatting
- May require many examples for complex tasks
- Higher inference cost due to long prompts

*Source: "Language Models are Few-Shot Learners" (Brown et al., 2020)*"""
    
    elif "training" in question_lower or "objective" in question_lower or "loss" in question_lower:
        answer = """**Training Objectives in Transformer Models**

**Transformer (Original Model):**

The original Transformer used different objectives for encoder and decoder:

1. **Machine Translation Objective**
   - Cross-entropy loss on target sequence
   - Teacher forcing during training
   - Given source sentence, predict target sentence word by word
   - Loss: -log P(y|x) where y is target, x is source

2. **Label Smoothing**
   - Regularization technique used during training
   - Softens hard targets (0s and 1s) to (ε, 1-ε)
   - Prevents overconfidence, improves generalization
   - Default ε = 0.1 in the paper

**GPT Models (Decoder-only):**

1. **Causal Language Modeling**
   - Predict next token given previous tokens
   - Autoregressive: P(x₁, x₂, ..., xₙ) = ∏ P(xᵢ|x₁, ..., xᵢ₋₁)
   - Loss: Negative log-likelihood of the sequence

2. **Unsupervised Pre-training**
   - Trained on massive text corpus (e.g., Common Crawl, WebText)
   - No human labels required
   - Learns language patterns, world knowledge, reasoning

**BERT Models (Encoder-only):**

1. **Masked Language Modeling (MLM)**
   - Randomly mask 15% of tokens in input
   - Predict original tokens based on context
   - Bidirectional context (can see both left and right)
   - Loss: Cross-entropy on masked positions only

2. **Next Sentence Prediction (NSP)**
   - Binary classification task
   - Given two sentences A and B, predict if B follows A
   - Helps learn sentence relationships
   - Later shown to be less important (removed in RoBERTa)

**RAG Models:**

1. **End-to-End Training**
   - Jointly train retriever and generator
   - Marginalizes over retrieved documents
   - Loss: Negative log-likelihood considering all retrieved docs

2. **Two Training Strategies**
   - **RAG-Sequence**: Same retrieved docs for entire sequence
   - **RAG-Token**: Can use different docs for each token

**Key Training Techniques:**

1. **Warmup and Learning Rate Scheduling**
   - Linear warmup for first N steps
   - Then decrease with inverse square root of step number
   - Critical for stable training

2. **Gradient Clipping**
   - Prevents exploding gradients
   - Typically clip to max norm of 1.0

3. **Dropout**
   - Applied to attention weights and feed-forward layers
   - Typical rate: 0.1
   - Prevents overfitting

4. **Layer Normalization**
   - Applied before or after each sub-layer
   - Stabilizes training, allows higher learning rates

*Sources: Vaswani et al. (2017), Brown et al. (2020), Lewis et al. (2020)*"""
    
    elif "scaling" in question_lower or "size" in question_lower or "parameter" in question_lower:
        answer = """**Scaling Laws in Language Models**

**GPT-3 Model Sizes:**

The GPT-3 family includes 8 different model sizes, demonstrating clear scaling trends:

| Model | Parameters | Layers | Hidden Size | Heads | Learning Rate |
|-------|-----------|--------|-------------|-------|---------------|
| GPT-3 Small | 125M | 12 | 768 | 12 | 6.0 × 10⁻⁴ |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 3.0 × 10⁻⁴ |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 2.5 × 10⁻⁴ |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 2.0 × 10⁻⁴ |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 1.6 × 10⁻⁴ |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 1.2 × 10⁻⁴ |
| GPT-3 13B | 13B | 40 | 5140 | 40 | 1.0 × 10⁻⁴ |
| **GPT-3 175B** | **175B** | **96** | **12288** | **96** | **0.6 × 10⁻⁴** |

**Key Scaling Observations:**

1. **Performance Improvements**
   - Larger models consistently outperform smaller ones
   - Improvement is smooth and predictable
   - Few-shot learning ability emerges more strongly at larger scales

2. **Compute-Performance Relationship**
   - Performance scales as a power law with compute
   - Doubling model size requires ~10x more compute
   - But provides consistent performance gains

3. **Data Requirements**
   - Larger models can effectively utilize more data
   - GPT-3 trained on ~300 billion tokens
   - No sign of saturation - could benefit from more data

4. **Few-Shot vs Fine-tuning Gap**
   - Smaller models: large gap between few-shot and fine-tuned performance
   - GPT-3 175B: few-shot approaches fine-tuned performance on some tasks
   - Suggests emergent meta-learning capabilities

**Computational Costs:**

Training GPT-3 175B:
- **Compute**: ~3.14 × 10²³ FLOPS
- **Time**: Several weeks on thousands of GPUs
- **Energy**: Estimated ~1,287 MWh
- **Cost**: Estimated $4.6-12 million

**Transformer Base vs Big:**

Original Transformer paper compared two sizes:

| Model | Parameters | d_model | d_ff | Heads | Layers |
|-------|-----------|---------|------|-------|--------|
| Base | 65M | 512 | 2048 | 8 | 6 |
| Big | 213M | 1024 | 4096 | 16 | 6 |

**Practical Implications:**

1. **Model Selection**
   - Larger models: better quality, higher cost
   - Smaller models: faster, cheaper, good for many tasks
   - Trade-off between performance and resources

2. **Efficiency Techniques**
   - Distillation: train small model to mimic large one
   - Pruning: remove unnecessary connections
   - Quantization: reduce precision (FP16, INT8)

3. **Deployment Considerations**
   - GPT-3 175B: ~350GB in FP16, requires multiple GPUs
   - Smaller variants: can run on single GPU or CPU
   - API access often more practical than self-hosting

**Scaling Beyond GPT-3:**

Recent trends (as of the paper):
- Models continue to scale (GPT-4, PaLM, etc.)
- Diminishing returns may eventually occur
- Focus shifting to data quality, architecture improvements

*Source: "Language Models are Few-Shot Learners" (Brown et al., 2020)*"""
    
    elif "dataset" in question_lower or "data" in question_lower or "corpus" in question_lower:
        answer = """**Training Datasets in Modern Language Models**

**GPT-3 Training Data:**

GPT-3 was trained on a diverse mixture of datasets totaling ~570GB of text:

1. **Common Crawl (filtered)** - 60% of training mix
   - 410 billion tokens
   - Web pages from 2016-2019
   - Filtered for quality using classifier trained on high-quality references

2. **WebText2** - 22% of training mix
   - 19 billion tokens
   - Web pages from Reddit links with ≥3 upvotes
   - Higher quality than raw Common Crawl

3. **Books1** - 8% of training mix
   - Internet-based books corpus
   - ~12 billion tokens

4. **Books2** - 8% of training mix
   - Another books corpus
   - ~55 billion tokens

5. **Wikipedia** - 3% of training mix
   - English Wikipedia (excluding lists, tables, headers)
   - ~3 billion tokens
   - High-quality, factual content

**Data Processing:**

1. **Quality Filtering**
   - Trained classifier on (WebText, Common Crawl) pairs
   - Removed low-quality documents
   - Fuzzy deduplication to prevent overfitting

2. **Document Sampling**
   - Not all data seen equally often
   - Higher-quality sources sampled more frequently
   - Prevents bias toward lower-quality bulk data

**Original Transformer (WMT 2014):**

1. **English-German Translation**
   - 4.5 million sentence pairs
   - 36M tokens (English), 36M tokens (German)
   - WMT 2014 training data

2. **English-French Translation**
   - 36 million sentence pairs
   - 32,000 word piece vocabulary

**RAG Model Data:**

1. **Knowledge Source**
   - Wikipedia (Dec 2018 dump)
   - 21 million passages
   - Each passage ~100 words

2. **Task-Specific Datasets**
   - Natural Questions (NQ)
   - TriviaQA
   - WebQuestions
   - CuratedTREC
   - Used for evaluation and fine-tuning

**Data Augmentation Techniques:**

1. **Back-Translation**
   - Translate to another language and back
   - Creates paraphrases for training

2. **Noise Injection**
   - Random token deletion
   - Random token swapping
   - Improves robustness

**Data Quality Insights:**

1. **Quality > Quantity**
   - Clean, high-quality data more valuable than large noisy datasets
   - GPT-3 filtered out 90% of Common Crawl

2. **Diversity Matters**
   - Multiple data sources prevent overfitting
   - Improves generalization

3. **Deduplication Critical**
   - Prevents memorization
   - Improves test performance
   - Reduces training time

**Ethical Considerations:**

1. **Bias in Training Data**
   - Internet text reflects societal biases
   - Models can perpetuate stereotypes
   - Mitigation through filtering and fine-tuning

2. **Privacy Concerns**
   - Training data may contain personal information
   - Need for careful filtering and anonymization

3. **Copyright Issues**
   - Books and web content may be copyrighted
   - Ongoing legal and ethical debates

*Sources: Brown et al. (2020), Vaswani et al. (2017), Lewis et al. (2020)*"""
    
    elif "hyperparameter" in question_lower or "learning rate" in question_lower or "batch" in question_lower:
        answer = """**Hyperparameters in Transformer Models**

**Original Transformer (Base Model):**

**Architecture:**
- Layers (N): 6 encoder + 6 decoder layers
- Model dimension (d_model): 512
- Feed-forward dimension (d_ff): 2048
- Attention heads (h): 8
- Attention dimensions: d_k = d_v = d_model/h = 64
- Maximum sequence length: 512 tokens
- Vocabulary size: ~37,000 tokens (BPE)

**Training:**
- Optimizer: Adam (β₁=0.9, β₂=0.98, ε=10⁻⁹)
- Learning rate schedule: Warmup + inverse square root decay
  ```
  lr = d_model⁻⁰·⁵ × min(step⁻⁰·⁵, step × warmup_steps⁻¹·⁵)
  ```
- Warmup steps: 4,000
- Batch size: ~25,000 source + target tokens per batch
- Dropout: 0.1 (applied to embeddings, attention, feed-forward)
- Label smoothing: ε_ls = 0.1
- Training steps: 100,000 (base), 300,000 (big)
- Hardware: 8 NVIDIA P100 GPUs
- Training time: 12 hours (base), 3.5 days (big)

**Transformer Big Model:**
- Layers: 6 encoder + 6 decoder
- d_model: 1024
- d_ff: 4096
- h: 16
- d_k = d_v = 64 (same as base!)
- Dropout: 0.3 (higher than base)

**GPT-3 (175B) Hyperparameters:**

**Architecture:**
- Layers: 96
- d_model: 12,288
- d_ff: 49,152 (4 × d_model)
- Heads: 96
- d_head: 128
- Context window: 2048 tokens
- Vocabulary: 50,257 tokens (BPE)

**Training:**
- Batch size: 3.2 million tokens
- Learning rate: 0.6 × 10⁻⁴
- Optimizer: Adam (β₁=0.9, β₂=0.95, ε=10⁻⁸)
- Weight decay: 0.1
- Gradient clipping: 1.0 (global norm)
- Learning rate schedule: Cosine decay to 10% of max
- Warmup: 375 million tokens
- Total training: 300 billion tokens
- Precision: Mixed precision (FP16/FP32)

**Critical Hyperparameter Choices:**

1. **Learning Rate**
   - Most important hyperparameter
   - Too high: training diverges
   - Too low: slow convergence, poor performance
   - Typically inversely proportional to model size

2. **Batch Size**
   - Larger batches: more stable gradients, better hardware utilization
   - Too large: may hurt generalization
   - GPT-3 used very large batches (millions of tokens)

3. **Warmup**
   - Critical for training stability
   - Prevents early-stage instability
   - Typical: 5-10% of total steps

4. **Dropout**
   - Regularization to prevent overfitting
   - Larger models often need less dropout
   - Applied to embeddings, attention, and FFN layers

5. **Layer Normalization Position**
   - Pre-LN (before sub-layer): more stable for large models
   - Post-LN (after sub-layer): used in original paper
   - Modern models prefer Pre-LN

**Model Size vs Hyperparameters:**

Key scaling patterns:
```
d_model ∝ √(parameters)
layers ∝ ⁴√(parameters)
learning_rate ∝ 1/√(d_model)
batch_size ∝ parameters
```

**Sensitivity Analysis:**

Most sensitive hyperparameters (ranked):
1. Learning rate
2. Model size (d_model, layers)
3. Batch size
4. Warmup steps
5. Dropout rate
6. Weight decay

Less sensitive:
- Adam betas (β₁, β₂)
- Epsilon value
- Gradient clipping threshold

**Practical Tips:**

1. **Start with Standard Values**
   - Use proven configurations (Transformer Base/Big)
   - Adjust gradually based on validation performance

2. **Grid Search vs Random Search**
   - Random search often more efficient
   - Focus on most important hyperparameters first

3. **Learning Rate Finder**
   - Gradually increase LR from very small value
   - Find where loss starts increasing
   - Use value before divergence point

4. **Early Stopping**
   - Monitor validation loss
   - Stop when no improvement for N steps
   - Prevents overfitting

*Sources: Vaswani et al. (2017), Brown et al. (2020)*"""
    
    else:
        answer = """**Information from AI Research Papers**

I'm here to help you understand concepts from AI research papers, particularly focusing on:

- **Transformer Architecture**: Self-attention mechanisms, encoder-decoder structure, positional encodings
- **RAG Systems**: Retrieval-augmented generation, document retrieval, answer generation
- **Language Models**: GPT-3, BERT, T5, and their training methodologies
- **Attention Mechanisms**: Multi-head attention, self-attention, cross-attention
- **Few-Shot Learning**: In-context learning, prompt engineering

Please ask a specific question about any of these topics, and I'll provide detailed information based on the research papers in the knowledge base.

**Example questions:**
- What are the main components of a RAG model?
- How does positional encoding work in Transformers?
- Explain multi-head attention and its benefits
- What is few-shot learning in GPT-3?
- Describe the encoder-decoder architecture"""
    
    # Enhanced mock sources with detailed attribution
    sources = {
        'texts': [
            """📄 **Source 1: Attention Is All You Need** (Vaswani et al., 2017)
            
Section 3.1: Multi-Head Attention
"Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."

Section 3.2: Position-wise Feed-Forward Networks
"In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically."

Citation: Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (pp. 5998-6008).""",
            
            """📄 **Source 2: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (Lewis et al., 2020)
            
Section 2: Methods
"RAG models use the input sequence x to retrieve text documents z and use them as additional context when generating the target sequence y. We marginalize over the latent documents with a top-K approximation."

Section 3.1: Retrieval
"We use a pre-trained bi-encoder from DPR to initialize our retriever and a pre-trained BART-large model to initialize our generator."

Citation: Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive nlp tasks. arXiv preprint arXiv:2005.11401.""",
            
            """📄 **Source 3: Language Models are Few-Shot Learners** (Brown et al., 2020)
            
Section 2.1: Model Architecture
"We use the same model and architecture as GPT-2, including the modified initialization, pre-normalization, and reversible tokenization described therein, with the exception that we use alternating dense and locally banded sparse attention patterns in the layers of the transformer."

Section 3.1: Few-Shot Learning
"For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model."

Citation: Brown, T. B., Mann, B., Ryder, N., et al. (2020). Language models are few-shot learners. arXiv preprint arXiv:2005.14165.""",
        ],
        'images': []  # Images not available in debug mode - use full mode with API for image analysis
    }
    
    return {
        'answer': answer,
        'sources': sources
    }


def validate_api_key(api_key: str, provider: str) -> bool:
    """
    Validate API key format
    
    Args:
        api_key: API key to validate
        provider: Provider name ("OpenAI" or "Google")
        
    Returns:
        True if key format appears valid
    """
    if not api_key or len(api_key) < 10:
        return False
    
    if provider == "OpenAI":
        # OpenAI keys typically start with "sk-"
        return api_key.startswith("sk-")
    elif provider == "Google":
        # Google API keys are typically 39 characters
        return len(api_key) > 20
    
    return True


def create_download_link(text: str, filename: str = "response.txt") -> str:
    """
    Create a download link for text content
    
    Args:
        text: Text content to download
        filename: Name of the file
        
    Returns:
        HTML download link
    """
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Response</a>'
