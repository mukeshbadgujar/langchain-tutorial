# 04 - Embeddings and Vector Stores

## Overview

Embeddings and vector stores are fundamental components for building semantic search and retrieval systems in LangChain. This document covers embedding generation, vector storage, and similarity search techniques.

## Understanding Embeddings

Embeddings are numerical representations of text that capture semantic meaning. They enable similarity comparisons and semantic search capabilities.

### What Are Embeddings?

```python
# Text -> Embedding (Vector)
text = "LangChain is a framework for building AI applications"
embedding = [0.1, -0.5, 0.8, ...]  # 1536-dimensional vector (OpenAI)
```

### Types of Embeddings

1. **Dense Embeddings**: Fixed-size vectors (e.g., 1536 dimensions)
2. **Sparse Embeddings**: Variable-size vectors with mostly zeros
3. **Contextual Embeddings**: Context-aware representations

## Embedding Providers

### OpenAI Embeddings

Most commonly used in the project:

```python
from langchain_openai import OpenAIEmbeddings

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Generate embedding for text
text = "What is LangChain?"
embedding_vector = embeddings.embed_query(text)
print(f"Embedding dimension: {len(embedding_vector)}")
```

### Ollama Embeddings

For local/open-source models:

```python
from langchain_ollama import OllamaEmbeddings

# Local embeddings
local_embeddings = OllamaEmbeddings(
    model="llama3",
    base_url="http://localhost:11434"
)

# Generate embedding
embedding = local_embeddings.embed_query("Sample text")
```

### Hugging Face Embeddings

From the project's embeddings examples:

```python
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize Hugging Face embeddings
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Generate embeddings
text = "This is a sample sentence"
embedding = hf_embeddings.embed_query(text)
```

### Custom Embeddings

```python
from langchain_core.embeddings import Embeddings
from typing import List

class CustomEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return [self.embed_query(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        # Your custom embedding logic here
        return [0.1] * 384  # Example: 384-dimensional vector
```

## Vector Stores

Vector stores efficiently store and retrieve embeddings for similarity search.

### FAISS (Facebook AI Similarity Search)

Most frequently used in the project:

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Create FAISS vector store
vectorstore = FAISS.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings()
)

# Save to disk
vectorstore.save_local("faiss_index")

# Load from disk
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings=OpenAIEmbeddings()
)
```

### Chroma

```python
from langchain_community.vectorstores import Chroma

# Create Chroma vector store
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# Persist to disk
vectorstore.persist()
```

### Pinecone

```python
from langchain_community.vectorstores import Pinecone
import pinecone

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

# Create vector store
vectorstore = Pinecone.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(),
    index_name="langchain-index"
)
```

## Similarity Search

### Basic Similarity Search

```python
# Search for similar documents
query = "What is LangGraph?"
similar_docs = vectorstore.similarity_search(
    query=query,
    k=4  # Return top 4 similar documents
)

# Print results
for doc in similar_docs:
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Source: {doc.metadata.get('source', 'unknown')}")
    print("---")
```

### Similarity Search with Scores

```python
# Search with similarity scores
results = vectorstore.similarity_search_with_score(
    query="LangChain tutorial",
    k=3
)

for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content[:100]}...")
    print("---")
```

### Maximum Marginal Relevance (MMR)

```python
# MMR search for diverse results
mmr_results = vectorstore.max_marginal_relevance_search(
    query="LangChain applications",
    k=5,
    fetch_k=20,  # Fetch 20 candidates
    lambda_mult=0.5  # Diversity parameter
)
```

## Retrievers

Retrievers provide a standardized interface for document retrieval.

### Basic Retriever

From the project's RAG implementations:

```python
# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Use retriever
docs = retriever.invoke("What is LangChain?")
```

### Configurable Retriever

```python
# Advanced retriever configuration
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)
```

### Multi-Vector Retriever

```python
from langchain.retrievers import MultiVectorRetriever
from langchain_community.storage import InMemoryStore

# Create multi-vector retriever
id_key = "doc_id"
store = InMemoryStore()

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
    search_kwargs={"k": 4}
)
```

## Practical Example: Building a Knowledge Base

Complete example from the project's RAG system:

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def build_knowledge_base():
    """Build a knowledge base from web sources"""
    
    # Step 1: Load documents
    urls = [
        "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
        "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
        "https://langchain-ai.github.io/langgraph/how-tos/map-reduce/"
    ]
    
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    # Step 2: Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    # Step 3: Create vector store
    vectorstore = FAISS.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings()
    )
    
    # Step 4: Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    return retriever, vectorstore

# Usage
retriever, vectorstore = build_knowledge_base()

# Test retrieval
results = retriever.invoke("What is LangGraph?")
for doc in results:
    print(f"Content: {doc.page_content[:200]}...")
    print("---")
```

## Advanced Vector Store Operations

### Filtering

```python
# Filter by metadata
filtered_results = vectorstore.similarity_search(
    query="LangChain",
    k=5,
    filter={"source": "documentation"}
)
```

### Updating Vector Store

```python
# Add new documents
new_docs = [Document(page_content="New information", metadata={"source": "update"})]
vectorstore.add_documents(new_docs)

# Delete documents (if supported)
vectorstore.delete(ids=["doc_id_1", "doc_id_2"])
```

### Batch Operations

```python
# Batch addition
batch_docs = [
    Document(page_content=f"Document {i}", metadata={"batch": "1"})
    for i in range(100)
]

vectorstore.add_documents(batch_docs)
```

## Performance Optimization

### Embedding Caching

```python
import pickle
from typing import Dict, List

class CachedEmbeddings:
    def __init__(self, embeddings, cache_file="embeddings_cache.pkl"):
        self.embeddings = embeddings
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, List[float]]:
        try:
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def embed_query(self, text: str) -> List[float]:
        if text in self.cache:
            return self.cache[text]
        
        embedding = self.embeddings.embed_query(text)
        self.cache[text] = embedding
        self._save_cache()
        return embedding
```

### Batch Embedding

```python
def batch_embed_documents(texts: List[str], embeddings, batch_size=50):
    """Embed documents in batches for efficiency"""
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings
```

## Vector Store Comparison

### Performance Characteristics

```python
import time
from typing import List, Dict, Any

def benchmark_vector_stores(doc_splits: List[Document]) -> Dict[str, Any]:
    """Benchmark different vector stores"""
    
    results = {}
    
    # Test FAISS
    start = time.time()
    faiss_store = FAISS.from_documents(doc_splits, OpenAIEmbeddings())
    faiss_time = time.time() - start
    
    # Test Chroma
    start = time.time()
    chroma_store = Chroma.from_documents(doc_splits, OpenAIEmbeddings())
    chroma_time = time.time() - start
    
    # Test search performance
    query = "test query"
    
    start = time.time()
    faiss_results = faiss_store.similarity_search(query, k=5)
    faiss_search_time = time.time() - start
    
    start = time.time()
    chroma_results = chroma_store.similarity_search(query, k=5)
    chroma_search_time = time.time() - start
    
    return {
        "faiss": {
            "creation_time": faiss_time,
            "search_time": faiss_search_time,
            "results_count": len(faiss_results)
        },
        "chroma": {
            "creation_time": chroma_time,
            "search_time": chroma_search_time,
            "results_count": len(chroma_results)
        }
    }
```

## Hybrid Search

### Combining Dense and Sparse Retrieval

```python
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def create_hybrid_retriever(documents):
    """Create a hybrid retriever combining dense and sparse search"""
    
    # Dense retriever (vector-based)
    vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Sparse retriever (keyword-based)
    sparse_retriever = BM25Retriever.from_documents(documents)
    sparse_retriever.k = 5
    
    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.7, 0.3]  # 70% dense, 30% sparse
    )
    
    return ensemble_retriever
```

## Retrieval Tools

### Converting Retrievers to Tools

From the project's agent implementations:

```python
from langchain.tools.retriever import create_retriever_tool

# Create retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "retriever_vector_db_blog",
    "Search and return information about LangGraph"
)

# Use in agent
tools = [retriever_tool]
```

### Multiple Retriever Tools

```python
# LangGraph retriever
langgraph_retriever = vectorstore_langgraph.as_retriever()
langgraph_tool = create_retriever_tool(
    langgraph_retriever,
    "langgraph_docs",
    "Search LangGraph documentation"
)

# LangChain retriever
langchain_retriever = vectorstore_langchain.as_retriever()
langchain_tool = create_retriever_tool(
    langchain_retriever,
    "langchain_docs",
    "Search LangChain documentation"
)

# Combine tools
all_tools = [langgraph_tool, langchain_tool]
```

## Monitoring and Debugging

### Embedding Quality Assessment

```python
def assess_embedding_quality(embeddings, test_queries):
    """Assess embedding quality with test queries"""
    
    results = {}
    
    for query in test_queries:
        # Get embedding
        embedding = embeddings.embed_query(query)
        
        # Calculate statistics
        results[query] = {
            "dimension": len(embedding),
            "norm": sum(x**2 for x in embedding)**0.5,
            "mean": sum(embedding) / len(embedding),
            "std": (sum((x - results[query]["mean"])**2 for x in embedding) / len(embedding))**0.5
        }
    
    return results
```

### Search Result Analysis

```python
def analyze_search_results(vectorstore, queries, k=5):
    """Analyze search result quality"""
    
    analysis = {}
    
    for query in queries:
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        scores = [score for _, score in results]
        content_lengths = [len(doc.page_content) for doc, _ in results]
        
        analysis[query] = {
            "result_count": len(results),
            "avg_score": sum(scores) / len(scores),
            "score_range": (min(scores), max(scores)),
            "avg_content_length": sum(content_lengths) / len(content_lengths)
        }
    
    return analysis
```

## Best Practices

### 1. Choose Appropriate Embedding Models

```python
# For general purposes
general_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# For code-specific tasks
code_embeddings = HuggingFaceEmbeddings(
    model_name="microsoft/codebert-base"
)

# For domain-specific tasks
domain_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### 2. Optimize Chunk Size for Embeddings

```python
def optimize_for_embeddings(documents, target_embedding_size=1536):
    """Optimize document chunks for embedding models"""
    
    # Calculate optimal chunk size
    # Rule of thumb: ~200-500 tokens per chunk for most models
    optimal_chunk_size = 800
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=optimal_chunk_size,
        chunk_overlap=100,
        length_function=len
    )
    
    return splitter.split_documents(documents)
```

### 3. Handle Embedding Errors

```python
def safe_embedding_creation(documents, embeddings):
    """Create embeddings with error handling"""
    
    successful_docs = []
    failed_docs = []
    
    for doc in documents:
        try:
            # Test embedding creation
            test_embedding = embeddings.embed_query(doc.page_content[:100])
            successful_docs.append(doc)
        except Exception as e:
            print(f"Failed to create embedding for document: {e}")
            failed_docs.append(doc)
    
    if successful_docs:
        vectorstore = FAISS.from_documents(successful_docs, embeddings)
        return vectorstore, failed_docs
    else:
        return None, failed_docs
```

### 4. Monitor Vector Store Performance

```python
import psutil
import time

def monitor_vector_store_performance(vectorstore, queries):
    """Monitor vector store performance"""
    
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    start_time = time.time()
    
    for query in queries:
        results = vectorstore.similarity_search(query, k=5)
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    return {
        "total_time": end_time - start_time,
        "avg_time_per_query": (end_time - start_time) / len(queries),
        "memory_usage": end_memory - start_memory,
        "queries_processed": len(queries)
    }
```

## Next Steps

Continue with:

- [05 - RAG Systems](05-rag-systems.md)
- [06 - LangGraph Fundamentals](06-langgraph-fundamentals.md)
- [07 - Chatbots and Conversational AI](07-chatbots-conversational-ai.md)

## Key Takeaways

- **Embeddings** convert text to numerical vectors for semantic search
- **Vector stores** efficiently store and retrieve embeddings
- **Retrievers** provide standardized interfaces for document retrieval
- **Similarity search** enables finding semantically similar content
- **Performance optimization** is crucial for production systems
- **Monitoring** helps maintain system quality
- **Hybrid approaches** combine multiple retrieval methods

---

**Remember**: The quality of your embeddings and vector store setup directly impacts the performance of your RAG and search systems. Choose appropriate models and configurations for your specific use case.
