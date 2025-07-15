# 05 - RAG (Retrieval-Augmented Generation) Systems

## Overview

RAG systems combine retrieval mechanisms with language models to provide accurate, context-aware responses based on external knowledge. This document covers various RAG implementations found in the project.

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that enhances language models by retrieving relevant information from external knowledge bases before generating responses.

### RAG Architecture

```
Query → Retriever → Relevant Documents → LLM → Response
  ↓         ↓            ↓              ↓        ↓
User    Vector Store  Context Docs   Generator  Answer
```

### Benefits of RAG

- **Up-to-date Information**: Access to current data beyond training cutoff
- **Factual Accuracy**: Grounded responses based on reliable sources
- **Transparency**: Ability to trace answers back to sources
- **Domain Expertise**: Specialized knowledge without retraining
- **Reduced Hallucinations**: Factual grounding reduces false information

## Basic RAG Implementation

### Simple RAG Chain

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_basic_rag():
    """Create a basic RAG system"""
    
    # Load and process documents
    urls = [
        "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
        "https://langchain-ai.github.io/langgraph/tutorials/workflows/"
    ]
    
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    # Create vector store
    vectorstore = FAISS.from_documents(doc_splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    # Create RAG chain
    template = """Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4")
    
    # Chain components
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain

# Usage
rag_chain = create_basic_rag()
response = rag_chain.invoke("What is LangGraph?")
print(response)
```

## Agentic RAG

From the project's `1-AgenticRAG.ipynb`, this approach uses agents to decide when and how to retrieve information.

### Agent-Based RAG System

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate

def create_agentic_rag():
    """Create an agentic RAG system"""
    
    # Set up multiple knowledge bases
    # LangGraph knowledge base
    langgraph_urls = [
        "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
        "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
        "https://langchain-ai.github.io/langgraph/how-tos/map-reduce/"
    ]
    
    langgraph_docs = [WebBaseLoader(url).load() for url in langgraph_urls]
    langgraph_docs_list = [item for sublist in langgraph_docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    langgraph_splits = text_splitter.split_documents(langgraph_docs_list)
    
    vectorstore_langgraph = FAISS.from_documents(langgraph_splits, OpenAIEmbeddings())
    retriever_langgraph = vectorstore_langgraph.as_retriever()
    
    # LangChain knowledge base
    langchain_urls = [
        "https://python.langchain.com/docs/tutorials/",
        "https://python.langchain.com/docs/tutorials/chatbot/",
        "https://python.langchain.com/docs/tutorials/qa_chat_history/"
    ]
    
    langchain_docs = [WebBaseLoader(url).load() for url in langchain_urls]
    langchain_docs_list = [item for sublist in langchain_docs for item in sublist]
    
    langchain_splits = text_splitter.split_documents(langchain_docs_list)
    
    vectorstore_langchain = FAISS.from_documents(langchain_splits, OpenAIEmbeddings())
    retriever_langchain = vectorstore_langchain.as_retriever()
    
    # Create retriever tools
    retriever_tool_langgraph = create_retriever_tool(
        retriever_langgraph,
        "retriever_langgraph",
        "Search and return information about LangGraph"
    )
    
    retriever_tool_langchain = create_retriever_tool(
        retriever_langchain,
        "retriever_langchain",
        "Search and return information about LangChain"
    )
    
    # Create agent
    tools = [retriever_tool_langgraph, retriever_tool_langchain]
    
    prompt = PromptTemplate.from_template("""
    Answer the following questions as best you can. You have access to the following tools:
    
    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Question: {input}
    Thought: {agent_scratchpad}
    """)
    
    llm = ChatOpenAI(model="gpt-4")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

# Usage
agentic_rag = create_agentic_rag()
response = agentic_rag.invoke({
    "input": "What is the difference between LangChain and LangGraph?"
})
```

## Corrective RAG

From `2-CorrectiveRAG.ipynb`, this approach evaluates and corrects retrieved information.

### Corrective RAG Implementation

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def create_corrective_rag():
    """Create a corrective RAG system with document grading"""
    
    # Document grader
    grade_prompt = ChatPromptTemplate.from_template("""
    You are a grader assessing relevance of a retrieved document to a user question.
    
    Here is the retrieved document:
    {document}
    
    Here is the user question:
    {question}
    
    If the document contains keywords related to the user question, grade it as relevant.
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant.
    
    {format_instructions}
    """)
    
    parser = PydanticOutputParser(pydantic_object=GradeDocuments)
    
    llm = ChatOpenAI(model="gpt-4")
    
    grader = grade_prompt | llm | parser
    
    # RAG chain with correction
    def corrective_rag_chain(question: str, retriever):
        # Retrieve documents
        docs = retriever.invoke(question)
        
        # Grade documents
        relevant_docs = []
        for doc in docs:
            grade = grader.invoke({
                "document": doc.page_content,
                "question": question,
                "format_instructions": parser.get_format_instructions()
            })
            
            if grade.binary_score.lower() == "yes":
                relevant_docs.append(doc)
        
        # If no relevant documents, perform web search
        if not relevant_docs:
            # Fallback to web search or alternative retrieval
            print("No relevant documents found, falling back to web search")
            # Implement web search fallback here
            relevant_docs = docs  # For now, use original docs
        
        # Generate answer with relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        answer_prompt = ChatPromptTemplate.from_template("""
        Answer the question based on the following context:
        
        {context}
        
        Question: {question}
        
        Answer:
        """)
        
        answer_chain = answer_prompt | llm | StrOutputParser()
        
        return answer_chain.invoke({
            "context": context,
            "question": question
        })
    
    return corrective_rag_chain

# Usage
corrective_rag = create_corrective_rag()
retriever = vectorstore.as_retriever()
response = corrective_rag("What is LangGraph?", retriever)
```

## Adaptive RAG

From `4-AdaptiveRAG.ipynb`, this approach adapts retrieval strategy based on query complexity.

### Adaptive RAG System

```python
from langchain_core.pydantic_v1 import BaseModel, Field

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    
    datasource: str = Field(
        description="Given a user question choose which datasource would be most relevant for answering their question",
        enum=["vectorstore", "llm", "web_search"]
    )

def create_adaptive_rag():
    """Create an adaptive RAG system"""
    
    # Query router
    route_prompt = ChatPromptTemplate.from_template("""
    You are an expert at routing a user question to the appropriate data source.
    
    Based on the question below, choose the most appropriate datasource:
    
    - vectorstore: for questions about LangChain, LangGraph, or related documentation
    - llm: for general questions that don't require specific documentation
    - web_search: for current events or information not in the knowledge base
    
    Question: {question}
    
    {format_instructions}
    """)
    
    parser = PydanticOutputParser(pydantic_object=RouteQuery)
    llm = ChatOpenAI(model="gpt-4")
    
    router = route_prompt | llm | parser
    
    def adaptive_rag_chain(question: str, retriever):
        # Route the question
        route = router.invoke({
            "question": question,
            "format_instructions": parser.get_format_instructions()
        })
        
        if route.datasource == "vectorstore":
            # Use vector store retrieval
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            answer_prompt = ChatPromptTemplate.from_template("""
            Answer based on the following context:
            
            {context}
            
            Question: {question}
            """)
            
            answer_chain = answer_prompt | llm | StrOutputParser()
            return answer_chain.invoke({
                "context": context,
                "question": question
            })
        
        elif route.datasource == "llm":
            # Use LLM directly
            direct_prompt = ChatPromptTemplate.from_template("""
            Answer the following question:
            
            {question}
            """)
            
            direct_chain = direct_prompt | llm | StrOutputParser()
            return direct_chain.invoke({"question": question})
        
        elif route.datasource == "web_search":
            # Use web search (implement with actual search tool)
            print("Routing to web search")
            # Implement web search here
            return "Web search result placeholder"
    
    return adaptive_rag_chain

# Usage
adaptive_rag = create_adaptive_rag()
response = adaptive_rag("What is LangGraph?", retriever)
```

## Multi-Modal RAG

### RAG with Multiple Content Types

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredImageLoader
)

def create_multimodal_rag():
    """Create a RAG system handling multiple content types"""
    
    # Load different content types
    text_docs = WebBaseLoader("https://example.com").load()
    pdf_docs = PyPDFLoader("document.pdf").load()
    csv_docs = CSVLoader("data.csv").load()
    
    # Process each type appropriately
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    all_docs = []
    
    # Process text documents
    for doc in text_docs:
        doc.metadata["content_type"] = "text"
        all_docs.append(doc)
    
    # Process PDF documents
    for doc in pdf_docs:
        doc.metadata["content_type"] = "pdf"
        all_docs.append(doc)
    
    # Process CSV documents
    for doc in csv_docs:
        doc.metadata["content_type"] = "structured"
        all_docs.append(doc)
    
    # Split all documents
    doc_splits = text_splitter.split_documents(all_docs)
    
    # Create vector store
    vectorstore = FAISS.from_documents(doc_splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    # Content-type aware prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the following context from different sources:
    
    {context}
    
    Question: {question}
    
    Consider the different content types when formulating your answer.
    """)
    
    model = ChatOpenAI(model="gpt-4")
    
    def format_docs(docs):
        formatted = []
        for doc in docs:
            content_type = doc.metadata.get("content_type", "unknown")
            formatted.append(f"[{content_type.upper()}] {doc.page_content}")
        return "\n\n".join(formatted)
    
    # RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain
```

## Conversational RAG

### RAG with Memory

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def create_conversational_rag():
    """Create a conversational RAG system with memory"""
    
    # Set up retriever
    vectorstore = FAISS.from_documents(doc_splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    # Contextualize question based on chat history
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given a chat history and the latest user question 
        which might reference context in the chat history, formulate a standalone question 
        which can be understood without the chat history."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    
    # Answer generation prompt
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer the user's question based on the following context:
        
        {context}"""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    
    llm = ChatOpenAI(model="gpt-4")
    
    # Create contextualized retriever
    contextualized_question = contextualize_prompt | llm | StrOutputParser()
    
    def contextualized_retriever(input_dict):
        if input_dict.get("chat_history"):
            contextualized_q = contextualized_question.invoke(input_dict)
            return retriever.invoke(contextualized_q)
        else:
            return retriever.invoke(input_dict["input"])
    
    # RAG chain
    rag_chain = (
        {
            "context": contextualized_retriever,
            "input": RunnablePassthrough(),
            "chat_history": RunnablePassthrough()
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )
    
    # Add memory
    store = {}
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    
    return conversational_rag

# Usage
conv_rag = create_conversational_rag()
config = {"configurable": {"session_id": "session1"}}

response1 = conv_rag.invoke(
    {"input": "What is LangGraph?"},
    config=config
)

response2 = conv_rag.invoke(
    {"input": "How does it differ from LangChain?"},
    config=config
)
```

## RAG Evaluation

### Evaluating RAG Performance

```python
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class RAGEvaluation(BaseModel):
    """Evaluation metrics for RAG systems"""
    
    relevance_score: float = Field(description="Relevance of retrieved documents (0-1)")
    accuracy_score: float = Field(description="Accuracy of generated answer (0-1)")
    completeness_score: float = Field(description="Completeness of answer (0-1)")
    faithfulness_score: float = Field(description="Faithfulness to source documents (0-1)")

def evaluate_rag_system(rag_chain, test_questions, expected_answers):
    """Evaluate RAG system performance"""
    
    evaluation_prompt = ChatPromptTemplate.from_template("""
    Evaluate the following RAG system output:
    
    Question: {question}
    Expected Answer: {expected_answer}
    Generated Answer: {generated_answer}
    Source Documents: {source_docs}
    
    Provide scores (0-1) for:
    - Relevance: How relevant are the retrieved documents?
    - Accuracy: How accurate is the generated answer?
    - Completeness: How complete is the answer?
    - Faithfulness: How faithful is the answer to the source documents?
    
    {format_instructions}
    """)
    
    parser = PydanticOutputParser(pydantic_object=RAGEvaluation)
    evaluator = evaluation_prompt | ChatOpenAI(model="gpt-4") | parser
    
    results = []
    
    for question, expected in zip(test_questions, expected_answers):
        # Get RAG response
        response = rag_chain.invoke(question)
        
        # Evaluate
        evaluation = evaluator.invoke({
            "question": question,
            "expected_answer": expected,
            "generated_answer": response,
            "source_docs": "Retrieved documents",  # Add actual source docs
            "format_instructions": parser.get_format_instructions()
        })
        
        results.append({
            "question": question,
            "response": response,
            "evaluation": evaluation
        })
    
    return results
```

## Production RAG Optimization

### Caching and Performance

```python
from functools import lru_cache
import hashlib

class OptimizedRAG:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = vectorstore.as_retriever()
        self.response_cache = {}
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        return hashlib.md5(query.encode()).hexdigest()
    
    @lru_cache(maxsize=100)
    def _cached_retrieval(self, query: str):
        """Cache retrieval results"""
        return self.retriever.invoke(query)
    
    def invoke(self, query: str) -> str:
        """Invoke RAG with caching"""
        
        cache_key = self._get_cache_key(query)
        
        # Check response cache
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Retrieve documents (with caching)
        docs = self._cached_retrieval(query)
        
        # Generate response
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = ChatPromptTemplate.from_template("""
        Context: {context}
        Question: {question}
        Answer:
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "question": query
        })
        
        # Cache response
        self.response_cache[cache_key] = response
        
        return response
```

## Error Handling and Fallbacks

### Robust RAG Implementation

```python
def create_robust_rag():
    """Create a robust RAG system with error handling"""
    
    def safe_rag_chain(question: str, retriever, max_retries=3):
        """RAG chain with error handling and fallbacks"""
        
        for attempt in range(max_retries):
            try:
                # Primary retrieval
                docs = retriever.invoke(question)
                
                if not docs:
                    # Fallback 1: Relaxed search
                    docs = retriever.invoke(question, search_kwargs={"k": 10})
                
                if not docs:
                    # Fallback 2: Direct LLM response
                    llm = ChatOpenAI(model="gpt-4")
                    fallback_prompt = ChatPromptTemplate.from_template("""
                    Answer the following question to the best of your ability:
                    
                    {question}
                    
                    Note: This response is generated without access to specific documentation.
                    """)
                    
                    fallback_chain = fallback_prompt | llm | StrOutputParser()
                    return fallback_chain.invoke({"question": question})
                
                # Normal RAG processing
                context = "\n\n".join([doc.page_content for doc in docs])
                
                prompt = ChatPromptTemplate.from_template("""
                Answer based on the following context:
                
                {context}
                
                Question: {question}
                """)
                
                chain = prompt | ChatOpenAI(model="gpt-4") | StrOutputParser()
                
                return chain.invoke({
                    "context": context,
                    "question": question
                })
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return f"Error: Unable to process question after {max_retries} attempts"
                time.sleep(2 ** attempt)  # Exponential backoff
    
    return safe_rag_chain
```

## Next Steps

Continue with:

- [06 - LangGraph Fundamentals](06-langgraph-fundamentals.md)
- [07 - Chatbots and Conversational AI](07-chatbots-conversational-ai.md)
- [08 - Agents and Tools](08-agents-tools.md)

## Key Takeaways

- **RAG combines retrieval with generation** for accurate, grounded responses
- **Agentic RAG** uses intelligent agents to decide retrieval strategies
- **Corrective RAG** evaluates and improves retrieval quality
- **Adaptive RAG** routes queries to appropriate data sources
- **Conversational RAG** maintains context across interactions
- **Evaluation** is crucial for measuring RAG system performance
- **Error handling** ensures robust production systems

---

**Remember**: RAG systems require careful tuning of retrieval parameters, prompt engineering, and evaluation metrics to achieve optimal performance in production environments.
