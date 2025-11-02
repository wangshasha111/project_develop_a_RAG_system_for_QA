#!/usr/bin/env python
# coding: utf-8

# # Multi-modal RAG System
# 
# Many documents contain a mixture of content types, including text, tables and images.
# 
# Yet, information captured in images is lost in most RAG applications.
# 
# With the emergence of multimodal LLMs, like [GPT-4o](https://openai.com/index/hello-gpt-4o/), it is worth considering how to utilize images in RAG Systems:
# 
# ![](https://i.imgur.com/wcCDT38.gif)
# 
# 
# ---
# 
# * Use a multimodal LLM (such as [GPT-4o](https://openai.com/index/hello-gpt-4o/), [LLaVA](https://llava.hliu.cc/), or [FUYU-8b](https://www.adept.ai/blog/fuyu-8b)) to produce text summaries from images and tables
# * Embed and retrieve image, table and text summaries with a reference to the raw images and tables for given queries
# * Pass raw images, tables and text chunks to a multimodal LLM for answer synthesis   
# 
# ---
# 
# 
# 
# * We will use [Unstructured](https://unstructured.io/) to parse images, text, and tables from documents (PDFs).
# * We will use the [multi-vector retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector) with [Chroma](https://www.trychroma.com/) and Redis to store raw text and images along with their summaries for retrieval.
# * We will use GPT-4o for both image summarization (for retrieval) as well as final answer synthesis from join review of images and texts (or tables).
# 
# 
# ___Created By: [Dipanjan (DJ)](https://www.linkedin.com/in/dipanjans/)___
# 
# 

# ## Install Dependencies

# In[ ]:


# pull out the python code


# In[1]:


get_ipython().system('pip install langchain==0.3.10')
get_ipython().system('pip install langchain-openai==0.2.12')
get_ipython().system('pip install langchain-community==0.3.11')
get_ipython().system('pip install langchain-chroma==0.1.4')
get_ipython().system('pip install redis==5.2.0')


# In[2]:


# to prevent 403 errors with unstructured.io till they fix it
# refer: https://github.com/Unstructured-IO/unstructured/issues/3795
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')


# In[3]:


# installed newer version of unstructured as older one has 403 errors
# still being investigated even if they pushed a fix
# refer: https://github.com/Unstructured-IO/unstructured/issues/3795 for more details
# this version is stable and should be good
get_ipython().system(' pip install "unstructured[pdf]==0.16.11"')


# In[1]:


# install OCR dependencies for unstructured
get_ipython().system('sudo apt-get install tesseract-ocr')
get_ipython().system('sudo apt-get install poppler-utils')


# In[3]:


get_ipython().system('pip install htmltabletomd==1.0.0')


# ## Data Loading & Processing
# 
# ### Partition PDF tables, text, and images
#   

# In[4]:


get_ipython().system('wget https://sgp.fas.org/crs/misc/IF10244.pdf')


# In[27]:


get_ipython().system('rm -rf ./figures')


# We run the loader function twice to first extract all tables and then load all text and chunk it.
# 
# Reason for this is in the latest version of unstructured, you lose the table elements if you chunk directly.
# Refer to: https://github.com/Unstructured-IO/unstructured/issues/3827 for more details
# 
# So till they push a fix, we will use this method. Once fixed we will update the following loader section.

# In[28]:


from langchain_community.document_loaders import UnstructuredPDFLoader

doc = './IF10244.pdf'
# Extract tables
# takes 1-2 min on Colab
loader = UnstructuredPDFLoader(file_path=doc,
                               strategy='hi_res',
                               extract_images_in_pdf=True,
                               infer_table_structure=True,
                               mode='elements',
                               image_output_dir_path='./figures')
data = loader.load()


# In[38]:


[doc.metadata['category'] for doc in data if doc.metadata['category'] == 'Table']


# In[39]:


tables = [doc for doc in data if doc.metadata['category'] == 'Table']


# In[42]:


# Chunk text and extract text content
loader = UnstructuredPDFLoader(file_path=doc,
                               strategy='hi_res',
                               extract_images_in_pdf=True,
                               infer_table_structure=True,
                               chunking_strategy="by_title", # section-based chunking
                               max_characters=4000, # max size of chunks
                               new_after_n_chars=4000, # preferred size of chunks
                               combine_text_under_n_chars=2000, # smaller chunks < 2000 chars will be combined into a larger chunk
                               mode='elements',
                               image_output_dir_path='./figures')
texts = loader.load()


# In[43]:


len(texts)


# In[44]:


data = texts + tables


# In[45]:


len(data)


# In[46]:


data[5]


# In[47]:


from IPython.display import HTML, display, Markdown


# In[48]:


print(data[5].page_content)


# In[49]:


data[5].metadata['text_as_html']


# In[51]:


display(Markdown(data[5].metadata['text_as_html']))


# Since unstructured extracts the text from the table without any borders, we can use the HTML text and put it directly in prompts (LLMs understand HTML tables well) or even better convert HTML tables to Markdown tables as below

# In[53]:


import htmltabletomd

md_table = htmltabletomd.convert_table(data[5].metadata['text_as_html'])
print(md_table)


# ## Separate Data into Text and Table Elements

# In[54]:


docs = []
tables = []

for doc in data:
    if doc.metadata['category'] == 'Table':
        tables.append(doc)
    elif doc.metadata['category'] == 'CompositeElement':
        docs.append(doc)

len(docs), len(tables)


# ### Convert HTML Tables to Markdown

# In[55]:


for table in tables:
    table.page_content = htmltabletomd.convert_table(table.metadata['text_as_html'])


# In[56]:


for table in tables:
    print(table.page_content)
    print()


# ### View Extracted Images

# In[57]:


get_ipython().system(' ls -l ./figures')


# In[58]:


from IPython.display import Image

Image('./figures/figure-1-1.jpg')


# In[59]:


Image('./figures/figure-1-2.jpg')


# In[60]:


Image('./figures/figure-1-3.jpg')


# ### Enter Open AI API Key

# In[61]:


from getpass import getpass

OPENAI_KEY = getpass('Enter Open AI API Key: ')


# ### Setup Environment Variables

# In[62]:


import os

os.environ['OPENAI_API_KEY'] = OPENAI_KEY


# ### Load Connection to LLM
# 
# Here we create a connection to ChatGPT to use later in our chains

# In[64]:


from langchain_openai import ChatOpenAI

chatgpt = ChatOpenAI(model_name='gpt-4o', temperature=0)


# 
# 
# ### Text and Table summaries
# 
# We will use GPT-4o to produce table and, text summaries.
# 
# Text summaries are advised if using large chunk sizes (e.g., as set above, we use 4k token chunks).
# 
# Summaries are used to retrieve raw tables and / or raw chunks of text.

# In[65]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# Prompt
prompt_text = """
You are an assistant tasked with summarizing tables and text particularly for semantic retrieval.
These summaries will be embedded and used to retrieve the raw text or table elements
Give a detailed summary of the table or text below that is well optimized for retrieval.
For any tables also add in a one line description of what the table is about besides the summary.
Do not add redundant words like Summary.
Just output the actual summary content.

Table or text chunk:
{element}
"""
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain
summarize_chain = (
                    {"element": RunnablePassthrough()}
                      |
                    prompt
                      |
                    chatgpt
                      |
                    StrOutputParser() # extracts the response as text and returns it as a string
)

# Initialize empty summaries
text_summaries = []
table_summaries = []

text_docs = [doc.page_content for doc in docs]
table_docs = [table.page_content for table in tables]

text_summaries = summarize_chain.batch(text_docs, {"max_concurrency": 5})
table_summaries = summarize_chain.batch(table_docs, {"max_concurrency": 5})

len(text_summaries), len(table_summaries)


# In[66]:


text_summaries[0]


# In[67]:


table_summaries[0]


# ### Image summaries
# 
# We will use [GPT-4o](https://openai.com/index/hello-gpt-4o/) to produce the image summaries.
# 
# * We pass base64 encoded images

# In[68]:


import base64
import os

from langchain_core.messages import HumanMessage


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """Make image summary"""
    chat = ChatOpenAI(model="gpt-4o", temperature=0)

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """

    # Store base64 encoded images
    img_base64_list = []

    # Store image summaries
    image_summaries = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval.
                Remember these images could potentially contain graphs, charts or tables also.
                These summaries will be embedded and used to retrieve the raw image for question answering.
                Give a detailed summary of the image that is well optimized for retrieval.
                Do not add additional words like Summary, This image represents, etc.
             """

    # Apply to images
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return img_base64_list, image_summaries


# Image summaries
IMG_PATH = './figures'
imgs_base64, image_summaries = generate_img_summaries(IMG_PATH)


# In[69]:


len(imgs_base64), len(image_summaries)


# In[70]:


display(Image('./figures/figure-1-2.jpg'))


# In[71]:


image_summaries[1]


# ## Multi-vector retriever
# 
# Use [multi-vector-retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector#summary) to index image (and / or text, table) summaries, but retrieve raw images (along with raw texts or tables).

# ### Download and Install Redis as a DocStore
# 
# You can use any other database or cache as a docstore to store the raw text, table and image elements

# In[72]:


get_ipython().run_cell_magic('sh', '', 'curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg\necho "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list\nsudo apt-get update  > /dev/null 2>&1\nsudo apt-get install redis-stack-server  > /dev/null 2>&1\nredis-stack-server --daemonize yes\n')


# ### Open AI Embedding Models
# 
# LangChain enables us to access Open AI embedding models which include the newest models: a smaller and highly efficient `text-embedding-3-small` model, and a larger and more powerful `text-embedding-3-large` model.

# In[73]:


from langchain_openai import OpenAIEmbeddings

# details here: https://openai.com/blog/new-embedding-models-and-api-updates
openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')


# In[74]:


docs[0]


# ### Add to vectorstore & docstore
# 
# Add raw docs and doc summaries to [Multi Vector Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector#summary):
# 
# * Store the raw texts, tables, and images in the `docstore` (here we are using Redis).
# * Store the texts, table summaries, and image summaries and their corresponding embeddings in the `vectorstore` (here we are using Chroma) for efficient semantic retrieval.
# * Connect them using a common `document_id`

# In[75]:


import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.storage import RedisStore
from langchain_community.utilities.redis import get_client
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def create_multi_vector_retriever(
    docstore, vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """


    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
    )

    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    # Add texts, tables, and images
    # Check that text_summaries is not empty before adding
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    # Check that table_summaries is not empty before adding
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    # Check that image_summaries is not empty before adding
    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever


# The vectorstore to use to index the summaries and their embeddings
chroma_db = Chroma(
    collection_name="mm_rag",
    embedding_function=openai_embed_model,
    collection_metadata={"hnsw:space": "cosine"},
)

# Initialize the storage layer - to store raw images, text and tables
client = get_client('redis://localhost:6379')
redis_store = RedisStore(client=client) # you can use filestore, memorystory, any other DB store also

# Create retriever
retriever_multi_vector = create_multi_vector_retriever(
    redis_store,
    chroma_db,
    text_summaries,
    text_docs,
    table_summaries,
    table_docs,
    image_summaries,
    imgs_base64,
)


# In[76]:


retriever_multi_vector


# ## Test Multimodal RAG Retriever
# 
# 

# In[77]:


from IPython.display import HTML, display, Image
from PIL import Image
import base64
from io import BytesIO

def plt_img_base64(img_base64):
    """Disply base64 encoded string as image"""
    # Decode the base64 string
    img_data = base64.b64decode(img_base64)
    # Create a BytesIO object
    img_buffer = BytesIO(img_data)
    # Open the image using PIL
    img = Image.open(img_buffer)
    display(img)


# ### Check Retrieval
# 
# Examine retrieval; we get back images and tables also that are relevant to our question.

# In[88]:


# Check retrieval
query = "Analyze the wildfires trend with acres burned over the years"
docs = retriever_multi_vector.invoke(query, limit=5)

# We get 3 docs
len(docs)


# In[89]:


docs


# In[90]:


plt_img_base64(docs[2])


# In[91]:


# Check retrieval
query = "Tell me about the percentage of residences burned by wildfires in 2022"
docs = retriever_multi_vector.invoke(query, limit=5)

# We get 4 docs
len(docs)


# In[92]:


docs


# ## Utilities to separate retrieved elements
# 
# We need to bin the retrieved doc(s) into the correct parts of the GPT-4o prompt template.
# 
# Here we need to have text, table elements as one set of inputs and image elements as the other set of inputs as both require separate prompts in GPT-4o.

# In[93]:


import re
import base64

def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content.decode('utf-8')
        else:
            doc = doc.decode('utf-8')
        if looks_like_base64(doc) and is_image_data(doc):
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}


# In[94]:


# Check retrieval
query = "Tell me detailed statistics of the top 5 years with largest wildfire acres burned"
docs = retriever_multi_vector.invoke(query, limit=5)

# We get 3 docs
len(docs)


# In[95]:


docs


# In[96]:


is_image_data(docs[2].decode('utf-8'))


# In[97]:


r = split_image_text_types(docs)
r


# In[98]:


plt_img_base64(r['images'][0])


# ## Multimodal RAG
# 
# ### Build End-to-End Multimodal RAG Pipeline
# 
# Now let's connect our retriever, prompt instructions and build a multimodal RAG chain

# In[99]:


from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage

def multimodal_prompt_function(data_dict):
    """
    Create a multimodal prompt with both text and image context.

    This function formats the provided context from `data_dict`, which contains
    text, tables, and base64-encoded images. It joins the text (with table) portions
    and prepares the image(s) in a base64-encoded format to be included in a message.

    The formatted text and images (context) along with the user question are used to
    construct a prompt for GPT-4o
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # Adding the text and tables for analysis
    text_message = {
        "type": "text",
        "text": (
            f"""You are an analyst tasked with understanding detailed information and trends
                from text documents, data tables, and charts and graphs in images.
                You will be given context information below which will be a mix of text, tables,
                and images usually of charts or graphs.
                Use this information to provide answers related to the user question.
                Analyze all the context information including tables, text and images to generate the answer.
                Do not make up answers, use the provided context documents below
                and answer the question to the best of your ability.

                User question:
                {data_dict['question']}

                Context documents:
                {formatted_texts}

                Answer:
            """
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


# Create RAG chain
multimodal_rag = (
        {
            "context": itemgetter('context'),
            "question": itemgetter('input'),
        }
            |
        RunnableLambda(multimodal_prompt_function)
            |
        chatgpt
            |
        StrOutputParser()
)

# Pass input query to retriever and get context document elements
retrieve_docs = (itemgetter('input')
                    |
                retriever_multi_vector
                    |
                RunnableLambda(split_image_text_types))

# Below, we chain `.assign` calls. This takes a dict and successively
# adds keys-- "context" and "answer"-- where the value for each key
# is determined by a Runnable (function or chain executing at runtime).
# This helps in also having the retrieved context along with the answer generated by GPT-4o
multimodal_rag_w_sources = (RunnablePassthrough.assign(context=retrieve_docs)
                                               .assign(answer=multimodal_rag)
)


# In[100]:


# Run RAG chain
query = "Tell me detailed statistics of the top 5 years with largest wildfire acres burned"
response = multimodal_rag_w_sources.invoke({'input': query})
response


# In[101]:


def multimodal_rag_qa(query):
    response = multimodal_rag_w_sources.invoke({'input': query})
    print('=='*50)
    print('Answer:')
    display(Markdown(response['answer']))
    print('--'*50)
    print('Sources:')
    text_sources = response['context']['texts']
    img_sources = response['context']['images']
    for text in text_sources:
        display(Markdown(text))
        print()
    for img in img_sources:
        plt_img_base64(img)
        print()
    print('=='*50)


# In[102]:


query = "Tell me detailed statistics of the top 5 years with largest wildfire acres burned"
multimodal_rag_qa(query)


# In[105]:


# Run RAG chain
query = "Analyze the wildfires trend with acres burned over the years"
multimodal_rag_qa(query)


# In[104]:


# Run RAG chain
query = "Tell me about the percentage of residences burned by wildfires in 2022"
multimodal_rag_qa(query)


# In[ ]:




