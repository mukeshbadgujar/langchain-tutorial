# 02 - LangChain Basics

## Core Components Deep Dive

This document explores the fundamental building blocks of LangChain applications, with practical examples from the project codebase.

## Language Models (LLMs)

Language models are the foundation of any LangChain application. They process text input and generate responses.

### Supported Providers

#### OpenAI Integration
```python
from langchain_openai import ChatOpenAI

# Initialize OpenAI model
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Basic invocation
result = llm.invoke("What is generative AI?")
print(result.content)
```

#### Groq Integration
```python
from langchain_groq import ChatGroq

# Initialize Groq model
model = ChatGroq(
    model="Gemma2-9b-It",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Invoke with messages
from langchain_core.messages import HumanMessage
result = model.invoke([
    HumanMessage(content="Explain LangChain in simple terms")
])
```

#### Ollama Integration
```python
from langchain_ollama import ChatOllama

# Local model with Ollama
local_model = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434"
)
```

### Model Configuration

```python
# Model with specific parameters
model = ChatGroq(
    model="Gemma2-9b-It",
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    groq_api_key=groq_api_key
)
```

## Message System

LangChain uses a structured message system for communication with LLMs.

### Message Types

```python
from langchain_core.messages import (
    HumanMessage,      # User input
    AIMessage,         # Assistant response
    SystemMessage,     # System instructions
    FunctionMessage    # Function call results
)

# Example conversation
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Hi, My name is Krish and I am a Chief AI Engineer"),
    AIMessage(content="Hello Krish! Nice to meet you. How can I help you today?"),
    HumanMessage(content="What's my name and what do I do?")
]

response = model.invoke(messages)
```

### Message History Example

From the project's chatbot implementation:

```python
from langchain_core.messages import HumanMessage, AIMessage

# Conversation with context
conversation = model.invoke([
    HumanMessage(content="Hi, My name is Krish and I am a Chief AI Engineer"),
    AIMessage(content="Hello Krish! It's nice to meet you..."),
    HumanMessage(content="Hey What's my name and what do I do?")
])
```

## Prompt Templates

Prompt templates provide a structured way to create reusable prompts with variables.

### Basic Prompt Template

```python
from langchain_core.prompts import ChatPromptTemplate

# Simple template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI Engineer. Provide answers based on questions."),
    ("user", "{input}")
])

# Format the prompt
formatted = prompt.format(input="What is LangChain?")
```

### Advanced Template with Multiple Variables

```python
# Translation template
translation_prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following into {language}:"),
    ("user", "{text}")
])

# Usage
result = translation_prompt.invoke({
    "language": "French",
    "text": "Hello, how are you?"
})
```

### Template from Messages

```python
# More complex template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Your task is to {task}."),
    ("user", "Context: {context}"),
    ("user", "Question: {question}")
])

# Invoke with multiple variables
response = prompt.invoke({
    "role": "data scientist",
    "task": "analyze the given data",
    "context": "Sales data from Q1 2024",
    "question": "What are the trends?"
})
```

## Output Parsers

Output parsers format and structure the raw LLM responses.

### String Output Parser

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# Parse AIMessage to string
result = model.invoke(messages)
parsed_result = parser.invoke(result)
print(type(parsed_result))  # <class 'str'>
```

### JSON Output Parser

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Define output schema
class PersonInfo(BaseModel):
    name: str = Field(description="person's name")
    age: int = Field(description="person's age")
    occupation: str = Field(description="person's job")

# Create parser
parser = JsonOutputParser(pydantic_object=PersonInfo)

# Update prompt to include format instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract person information from the text."),
    ("user", "{text}"),
    ("user", "Format: {format_instructions}")
])

# Chain with parser
chain = prompt | model | parser
result = chain.invoke({
    "text": "John is 30 years old and works as a software engineer",
    "format_instructions": parser.get_format_instructions()
})
```

## LangChain Expression Language (LCEL)

LCEL provides a declarative way to compose chains.

### Basic Chain

```python
# Simple chain: prompt -> model -> parser
chain = prompt | model | parser

# Execute chain
response = chain.invoke({"input": "Tell me about AI"})
```

### Complex Chain with Multiple Steps

```python
from langchain_core.runnables import RunnableLambda

# Custom processing function
def process_input(input_data):
    return {"processed": input_data.upper()}

# Multi-step chain
complex_chain = (
    RunnableLambda(process_input) |
    prompt |
    model |
    parser
)
```

### Parallel Processing

```python
from langchain_core.runnables import RunnableParallel

# Parallel execution
parallel_chain = RunnableParallel({
    "translation": translation_prompt | model | parser,
    "summary": summary_prompt | model | parser
})

result = parallel_chain.invoke({
    "text": "Your input text here",
    "language": "Spanish"
})
```

## Memory and Session Management

Memory allows LangChain applications to maintain conversation context.

### Chat Message History

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Session store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Add memory to chain
with_message_history = RunnableWithMessageHistory(
    model,
    get_session_history
)
```

### Using Memory in Conversations

```python
# Configuration for session
config = {"configurable": {"session_id": "chat1"}}

# First message
response1 = with_message_history.invoke(
    [HumanMessage(content="Hi, My name is Krish")],
    config=config
)

# Second message (with memory)
response2 = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config
)
# Should remember "Krish"
```

### Multiple Sessions

```python
# Different session
config2 = {"configurable": {"session_id": "chat2"}}

response3 = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config2
)
# Won't remember "Krish" - different session
```

## Practical Examples from the Project

### 1. Simple LLM Application

From `LCEL/simplellmLCEL.ipynb`:

```python
# Translation application
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# Setup
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
parser = StrOutputParser()

# Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following into {language}:"),
    ("user", "{text}")
])

# Chain
chain = prompt | model | parser

# Execute
result = chain.invoke({
    "language": "French",
    "text": "Hello, how are you?"
})
```

### 2. Chatbot with Memory

From `1-chatbots.ipynb`:

```python
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Setup
load_dotenv()
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=os.getenv("GROQ_API_KEY"))

# Memory
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Chatbot with memory
chatbot = RunnableWithMessageHistory(model, get_session_history)

# Configuration
config = {"configurable": {"session_id": "user123"}}

# Conversation
response = chatbot.invoke(
    [HumanMessage(content="Hi, I'm John")],
    config=config
)

# Later in conversation
response = chatbot.invoke(
    [HumanMessage(content="What's my name?")],
    config=config
)
```

## Data Validation with Pydantic

From `intro.ipynb` and `4-pydantic.ipynb`:

### Basic Pydantic Model

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    city: str

# Usage
person = Person(name="Krish", age=35, city="Bangalore")
print(person)
```

### LangChain with Pydantic

```python
from typing import Optional
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    name: str = Field(description="User's full name")
    age: Optional[int] = Field(description="User's age", default=None)
    interests: list[str] = Field(description="List of interests")

# Use in chain
def parse_user_info(text: str) -> UserProfile:
    # Extract information from text
    # Return structured data
    pass
```

## Error Handling and Best Practices

### Robust Error Handling

```python
def safe_invoke(chain, input_data):
    try:
        result = chain.invoke(input_data)
        return result
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Usage
result = safe_invoke(chain, {"input": "test"})
if result:
    print(result)
else:
    print("Failed to process request")
```

### Retry Logic

```python
import time

def invoke_with_retry(chain, input_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # Exponential backoff
    return None
```

## Advanced Chain Patterns

### Conditional Chain

```python
from langchain_core.runnables import RunnableBranch

# Conditional logic
def route_question(input_data):
    question = input_data["question"].lower()
    if "weather" in question:
        return "weather_chain"
    elif "news" in question:
        return "news_chain"
    else:
        return "general_chain"

# Branch chain
branch = RunnableBranch(
    (lambda x: "weather" in x["question"].lower(), weather_chain),
    (lambda x: "news" in x["question"].lower(), news_chain),
    general_chain  # default
)
```

### Map-Reduce Pattern

```python
from langchain_core.runnables import RunnableMap

# Process multiple inputs
map_chain = RunnableMap({
    "summary": summary_chain,
    "sentiment": sentiment_chain,
    "keywords": keyword_chain
})

# Execute
result = map_chain.invoke({"text": "Your input text"})
```

## Testing and Debugging

### Unit Testing Chains

```python
import unittest

class TestChains(unittest.TestCase):
    def setUp(self):
        self.chain = prompt | model | parser
    
    def test_basic_response(self):
        result = self.chain.invoke({"input": "test"})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
```

### Debugging with LangSmith

```python
import os

# Enable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my_project"

# Your chain will now be traced
result = chain.invoke({"input": "test"})
```

## Performance Optimization

### Caching

```python
from langchain_core.caches import InMemoryCache

# Enable caching
from langchain_core.globals import set_llm_cache
set_llm_cache(InMemoryCache())

# Subsequent identical requests will be cached
```

### Streaming

```python
# Streaming responses
for chunk in chain.stream({"input": "Tell me a story"}):
    print(chunk, end="", flush=True)
```

## Next Steps

Continue your learning journey with:

- [03 - Data Ingestion and Processing](03-data-ingestion-processing.md)
- [04 - Embeddings and Vector Stores](04-embeddings-vectorstores.md)
- [05 - RAG Systems](05-rag-systems.md)

## Key Takeaways

- **LLMs** are the core processing units in LangChain
- **Prompt templates** provide structured, reusable prompts
- **Output parsers** format and validate responses
- **LCEL** enables declarative chain composition
- **Memory** maintains conversation context
- **Error handling** is crucial for production applications
- **Testing and debugging** ensure reliability

---

**Remember**: Start simple and gradually build complexity. Each component should be thoroughly tested before combining into larger systems.
