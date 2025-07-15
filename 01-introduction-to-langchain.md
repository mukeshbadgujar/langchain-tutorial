# 01 - Introduction to LangChain

## What is LangChain?

LangChain is a powerful framework for developing applications powered by large language models (LLMs). It provides a standardized interface for working with different LLM providers, memory management, and building complex workflows.

## Key Features

- **Model Agnostic**: Works with multiple LLM providers (OpenAI, Groq, Ollama, etc.)
- **Memory Management**: Maintains conversation history and context
- **Chain Operations**: Combine multiple operations in sequence
- **Tool Integration**: Connect LLMs with external tools and APIs
- **Vector Database Support**: Built-in support for similarity search
- **Streaming Support**: Real-time response streaming

## Core Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Input        │    │   LangChain     │    │    Output       │
│   (Prompt)      │───▶│   Processing    │───▶│   (Response)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   LLM Provider  │
                    │ (OpenAI, Groq)  │
                    └─────────────────┘
```

## Installation and Setup

### Basic Installation

```bash
pip install langchain
pip install langchain-core
pip install langchain-community
```

### Provider-Specific Packages

```bash
# OpenAI
pip install langchain-openai

# Groq
pip install langchain-groq

# Additional dependencies
pip install python-dotenv
pip install faiss-cpu
```

### Environment Configuration

Create a `.env` file:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Groq Configuration
GROQ_API_KEY=your_groq_api_key_here

# LangSmith for tracking (optional)
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=your_project_name
```

## Basic Usage Example

Here's a simple example from the project showing how to get started:

```python
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Initialize the model
model = ChatGroq(
    model="Gemma2-9b-It",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Simple invocation
result = model.invoke([
    HumanMessage(content="What is generative AI?")
])

print(result.content)
```

## Key Concepts

### 1. Language Models (LLMs)

Language models are the core of LangChain applications. They process text input and generate responses.

```python
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# OpenAI model
openai_model = ChatOpenAI(model="gpt-4")

# Groq model
groq_model = ChatGroq(model="Gemma2-9b-It")
```

### 2. Messages

LangChain uses a message-based system for communication:

```python
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi there! How can I help you?"),
    HumanMessage(content="What's the weather like?")
]
```

### 3. Prompt Templates

Templates make it easy to create reusable prompts:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI Engineer. Provide answers based on questions."),
    ("user", "{input}")
])
```

### 4. Output Parsers

Parse and format model outputs:

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
result = parser.invoke(model_response)
```

### 5. Chains (LCEL)

Chain components together using LangChain Expression Language:

```python
# Simple chain
chain = prompt | model | parser

# Execute chain
response = chain.invoke({"input": "Tell me about LangChain"})
```

## Project Structure Overview

Based on the project analysis, here's how LangChain is organized:

```
1-Basics+Of+Langchain/
├── 1.1-openai/              # OpenAI integration
├── 1.2-ollama/              # Ollama integration
├── 3.2-DataIngestion/       # Data loading
├── 3.3-Data Transformer/    # Data processing
├── 4-Embeddings/            # Vector embeddings
└── 5-VectorStore/           # Vector storage
```

## Memory and Session Management

LangChain provides memory capabilities for maintaining conversation context:

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Create message store
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Add memory to chain
with_message_history = RunnableWithMessageHistory(
    model,
    get_session_history
)
```

## Common Use Cases

### 1. Chatbots
Build conversational AI with memory and context awareness.

### 2. Question Answering
Create systems that can answer questions based on documents.

### 3. Content Generation
Generate articles, blogs, and creative content.

### 4. Data Analysis
Process and analyze large datasets with natural language queries.

### 5. Workflow Automation
Automate complex business processes with AI assistance.

## Best Practices

### 1. Error Handling
Always implement proper error handling:

```python
try:
    result = chain.invoke({"input": user_input})
except Exception as e:
    print(f"Error: {e}")
    # Handle gracefully
```

### 2. API Key Management
Use environment variables for API keys:

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

### 3. Model Selection
Choose appropriate models for your use case:
- **GPT-4**: Best quality, higher cost
- **Gemma2-9b**: Good balance of speed and quality
- **Llama3**: Open-source alternative

### 4. Monitoring and Debugging
Use LangSmith for tracking:

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my_project"
```

## Real-World Example: Simple Chatbot

Here's a complete example from the project:

```python
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Setup
load_dotenv()

# Initialize model
model = ChatGroq(
    model="Gemma2-9b-It",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Memory setup
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Create chatbot with memory
chatbot = RunnableWithMessageHistory(model, get_session_history)

# Configuration
config = {"configurable": {"session_id": "user123"}}

# Conversation
response1 = chatbot.invoke(
    [HumanMessage(content="Hi, I'm John")],
    config=config
)

response2 = chatbot.invoke(
    [HumanMessage(content="What's my name?")],
    config=config
)

print(response2.content)  # Should remember the name is John
```

## Next Steps

Now that you understand the basics of LangChain, proceed to:
- [02 - LangChain Basics](02-langchain-basics.md) for deeper component understanding
- [03 - Data Ingestion and Processing](03-data-ingestion-processing.md) for data handling
- [06 - LangGraph Fundamentals](06-langgraph-fundamentals.md) for workflow orchestration

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangSmith for Debugging](https://smith.langchain.com/)
- [Community Examples](https://github.com/langchain-ai/langchain)

---

**Key Takeaways:**
- LangChain provides a unified interface for LLM applications
- Components can be chained together using LCEL
- Memory and session management enable conversational AI
- Proper setup and configuration are crucial for success
- Always implement error handling and monitoring
