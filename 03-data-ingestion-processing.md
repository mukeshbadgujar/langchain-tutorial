# 03 - Data Ingestion and Processing

## Overview

Data ingestion and processing are crucial steps in building LangChain applications. This document covers document loaders, text splitters, and data transformation techniques used throughout the project.

## Document Loaders

Document loaders are responsible for loading data from various sources into LangChain-compatible formats.

### Web-Based Loading

From the project's RAG implementations:

```python
from langchain_community.document_loaders import WebBaseLoader

# Load from URLs
urls = [
    "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
    "https://langchain-ai.github.io/langgraph/how-tos/map-reduce/"
]

# Load documents
docs = [WebBaseLoader(url).load() for url in urls]
```

### Multiple URL Loading

```python
# LangChain documentation URLs
langchain_urls = [
    "https://python.langchain.com/docs/tutorials/",
    "https://python.langchain.com/docs/tutorials/chatbot/",
    "https://python.langchain.com/docs/tutorials/qa_chat_history/"
]

# Load all documents
docs = [WebBaseLoader(url).load() for url in langchain_urls]

# Flatten the list of documents
docs_list = [item for sublist in docs for item in sublist]
```

### File-Based Loading

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredHTMLLoader
)

# Text files
text_loader = TextLoader("document.txt")
text_docs = text_loader.load()

# PDF files
pdf_loader = PyPDFLoader("document.pdf")
pdf_docs = pdf_loader.load()

# CSV files
csv_loader = CSVLoader("data.csv")
csv_docs = csv_loader.load()

# HTML files
html_loader = UnstructuredHTMLLoader("webpage.html")
html_docs = html_loader.load()
```

### Directory Loading

```python
from langchain_community.document_loaders import DirectoryLoader

# Load all text files from a directory
loader = DirectoryLoader(
    path="./documents",
    glob="**/*.txt",
    loader_cls=TextLoader
)
docs = loader.load()
```

## Text Splitters

Text splitters break large documents into smaller chunks for processing.

### Recursive Character Text Splitter

The most commonly used splitter in the project:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Maximum chunk size
    chunk_overlap=100,      # Overlap between chunks
    length_function=len,    # Function to measure length
    is_separator_regex=False
)

# Split documents
doc_splits = text_splitter.split_documents(docs_list)
```

### Character-Based Splitting

```python
from langchain_text_splitters import CharacterTextSplitter

# Split by character count
char_splitter = CharacterTextSplitter(
    separator="\n\n",       # Split on double newlines
    chunk_size=1000,
    chunk_overlap=200
)

chunks = char_splitter.split_documents(documents)
```

### Token-Based Splitting

```python
from langchain_text_splitters import TokenTextSplitter

# Split by token count
token_splitter = TokenTextSplitter(
    chunk_size=100,         # 100 tokens per chunk
    chunk_overlap=20
)

token_chunks = token_splitter.split_documents(documents)
```

### Code-Specific Splitting

```python
from langchain_text_splitters import (
    PythonCodeTextSplitter,
    JavaScriptTextSplitter,
    MarkdownTextSplitter
)

# Python code splitter
python_splitter = PythonCodeTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# JavaScript code splitter
js_splitter = JavaScriptTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# Markdown splitter
md_splitter = MarkdownTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
```

## Data Transformation

### Document Preprocessing

```python
def preprocess_documents(docs):
    """Clean and preprocess documents"""
    processed_docs = []
    
    for doc in docs:
        # Clean text
        content = doc.page_content
        content = content.strip()
        content = content.replace('\n\n', '\n')
        
        # Update metadata
        doc.page_content = content
        doc.metadata['processed'] = True
        
        processed_docs.append(doc)
    
    return processed_docs
```

### Metadata Enhancement

```python
def enhance_metadata(docs, source_type="web"):
    """Add additional metadata to documents"""
    
    for doc in docs:
        doc.metadata.update({
            'source_type': source_type,
            'processed_at': datetime.now().isoformat(),
            'chunk_length': len(doc.page_content),
            'word_count': len(doc.page_content.split())
        })
    
    return docs
```

### Content Filtering

```python
def filter_documents(docs, min_length=100, max_length=10000):
    """Filter documents by content length"""
    
    filtered_docs = []
    
    for doc in docs:
        content_length = len(doc.page_content)
        
        if min_length <= content_length <= max_length:
            filtered_docs.append(doc)
    
    return filtered_docs
```

## Advanced Data Processing

### Batch Processing

```python
from typing import List
from langchain_core.documents import Document

def process_documents_batch(urls: List[str], batch_size: int = 10):
    """Process URLs in batches"""
    
    all_docs = []
    
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i + batch_size]
        
        # Load batch
        batch_docs = [WebBaseLoader(url).load() for url in batch_urls]
        batch_docs = [item for sublist in batch_docs for item in sublist]
        
        # Process batch
        processed_batch = preprocess_documents(batch_docs)
        all_docs.extend(processed_batch)
    
    return all_docs
```

### Parallel Processing

```python
import concurrent.futures
from functools import partial

def load_url(url):
    """Load a single URL"""
    try:
        return WebBaseLoader(url).load()
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return []

def parallel_document_loading(urls, max_workers=5):
    """Load documents in parallel"""
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {executor.submit(load_url, url): url for url in urls}
        
        all_docs = []
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                docs = future.result()
                all_docs.extend(docs)
            except Exception as e:
                print(f"Error processing {url}: {e}")
    
    return all_docs
```

## Data Validation

### Document Validation

```python
from pydantic import BaseModel, validator
from typing import Optional

class DocumentValidator(BaseModel):
    content: str
    source: str
    metadata: dict
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v
    
    @validator('content')
    def content_length_check(cls, v):
        if len(v) > 100000:  # 100k character limit
            raise ValueError('Content too long')
        return v

def validate_documents(docs):
    """Validate document format and content"""
    
    validated_docs = []
    
    for doc in docs:
        try:
            validator = DocumentValidator(
                content=doc.page_content,
                source=doc.metadata.get('source', 'unknown'),
                metadata=doc.metadata
            )
            validated_docs.append(doc)
        except Exception as e:
            print(f"Document validation failed: {e}")
    
    return validated_docs
```

## Error Handling and Retry Logic

### Robust Document Loading

```python
import time
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_robust_loader(max_retries=3):
    """Create a web loader with retry logic"""
    
    def load_with_retry(url):
        for attempt in range(max_retries):
            try:
                loader = WebBaseLoader(url)
                return loader.load()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to load {url} after {max_retries} attempts: {e}")
                    return []
                time.sleep(2 ** attempt)  # Exponential backoff
        return []
    
    return load_with_retry
```

### Error Recovery

```python
def safe_document_processing(docs, processor_func):
    """Process documents with error recovery"""
    
    processed_docs = []
    failed_docs = []
    
    for doc in docs:
        try:
            processed_doc = processor_func(doc)
            processed_docs.append(processed_doc)
        except Exception as e:
            print(f"Processing failed for document: {e}")
            failed_docs.append(doc)
    
    return processed_docs, failed_docs
```

## Practical Example: Complete Pipeline

Here's a complete data processing pipeline from the project:

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def create_knowledge_base(urls):
    """Create a complete knowledge base from URLs"""
    
    # Step 1: Load documents
    print("Loading documents...")
    docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            url_docs = loader.load()
            docs.extend(url_docs)
        except Exception as e:
            print(f"Error loading {url}: {e}")
    
    # Step 2: Flatten document list
    docs_list = [item for sublist in docs for item in sublist]
    
    # Step 3: Split documents
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    # Step 4: Create vector store
    print("Creating vector store...")
    vectorstore = FAISS.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings()
    )
    
    # Step 5: Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    return retriever, vectorstore

# Usage
urls = [
    "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    "https://langchain-ai.github.io/langgraph/tutorials/workflows/"
]

retriever, vectorstore = create_knowledge_base(urls)
```

## Performance Optimization

### Efficient Text Splitting

```python
def optimize_text_splitting(docs, target_chunk_size=1000):
    """Optimize text splitting based on content analysis"""
    
    # Analyze document characteristics
    avg_paragraph_length = sum(
        len(p) for doc in docs 
        for p in doc.page_content.split('\n\n')
    ) / sum(
        len(doc.page_content.split('\n\n')) for doc in docs
    )
    
    # Adjust chunk size based on content
    if avg_paragraph_length > target_chunk_size:
        chunk_size = target_chunk_size
        chunk_overlap = int(target_chunk_size * 0.2)
    else:
        chunk_size = int(avg_paragraph_length * 2)
        chunk_overlap = int(avg_paragraph_length * 0.1)
    
    # Create optimized splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return splitter.split_documents(docs)
```

### Memory-Efficient Processing

```python
def process_large_documents(file_path, chunk_size=1000):
    """Process large documents without loading everything into memory"""
    
    def document_generator():
        with open(file_path, 'r', encoding='utf-8') as file:
            buffer = ""
            for line in file:
                buffer += line
                if len(buffer) >= chunk_size:
                    yield buffer
                    buffer = ""
            if buffer:
                yield buffer
    
    # Process in chunks
    for chunk in document_generator():
        # Process each chunk
        yield process_chunk(chunk)
```

## Integration with Vector Stores

### Preparing Documents for Vector Storage

```python
def prepare_for_vectorization(docs):
    """Prepare documents for vector storage"""
    
    prepared_docs = []
    
    for doc in docs:
        # Clean content
        content = doc.page_content.strip()
        
        # Ensure minimum content length
        if len(content) < 50:
            continue
        
        # Add processing metadata
        doc.metadata.update({
            'content_length': len(content),
            'processed_for_vector': True,
            'chunk_id': len(prepared_docs)
        })
        
        prepared_docs.append(doc)
    
    return prepared_docs
```

## Monitoring and Logging

### Processing Statistics

```python
import logging
from datetime import datetime

def log_processing_stats(docs, processed_docs):
    """Log processing statistics"""
    
    original_count = len(docs)
    processed_count = len(processed_docs)
    
    logging.info(f"Processing completed at {datetime.now()}")
    logging.info(f"Original documents: {original_count}")
    logging.info(f"Processed documents: {processed_count}")
    logging.info(f"Success rate: {processed_count/original_count*100:.2f}%")
    
    # Log content statistics
    total_content = sum(len(doc.page_content) for doc in processed_docs)
    avg_content_length = total_content / processed_count if processed_count > 0 else 0
    
    logging.info(f"Total content length: {total_content}")
    logging.info(f"Average content length: {avg_content_length:.2f}")
```

## Best Practices

### 1. Choose Appropriate Chunk Sizes

```python
# For dense technical content
technical_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

# For narrative content
narrative_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)
```

### 2. Preserve Context

```python
# Maintain context across chunks
def context_aware_splitting(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,  # Higher overlap for context
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return splitter.split_documents(docs)
```

### 3. Handle Different Content Types

```python
def adaptive_processing(docs):
    """Process different content types appropriately"""
    
    processed_docs = []
    
    for doc in docs:
        source_type = doc.metadata.get('source_type', 'unknown')
        
        if source_type == 'code':
            # Use code-specific splitter
            splitter = PythonCodeTextSplitter(chunk_size=1000)
        elif source_type == 'markdown':
            # Use markdown-specific splitter
            splitter = MarkdownTextSplitter(chunk_size=1000)
        else:
            # Use general splitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        
        splits = splitter.split_documents([doc])
        processed_docs.extend(splits)
    
    return processed_docs
```

## Next Steps

Continue with:

- [04 - Embeddings and Vector Stores](04-embeddings-vectorstores.md)
- [05 - RAG Systems](05-rag-systems.md)
- [06 - LangGraph Fundamentals](06-langgraph-fundamentals.md)

## Key Takeaways

- **Document loaders** handle various data sources
- **Text splitters** break content into manageable chunks
- **Preprocessing** improves data quality
- **Error handling** ensures robust processing
- **Performance optimization** is crucial for large datasets
- **Content type awareness** improves processing quality
- **Monitoring** helps track processing success

---

**Remember**: Quality data processing is the foundation of successful LangChain applications. Invest time in understanding your data and choosing appropriate processing strategies.
