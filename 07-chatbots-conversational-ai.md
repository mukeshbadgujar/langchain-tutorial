# 07 - Chatbots and Conversational AI

## Overview

This document covers building sophisticated chatbots using LangChain and LangGraph, focusing on memory management, conversation flow, and context awareness based on implementations in the project.

## Basic Chatbot Architecture

### Core Components

1. **Language Model**: The AI engine that generates responses
2. **Memory**: Stores conversation history
3. **Prompt Templates**: Structure the conversation context
4. **Session Management**: Handles multiple users/conversations

### Simple Chatbot Implementation

From `1-chatbots.ipynb`:

```python
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# Setup
load_dotenv()
model = ChatGroq(
    model="Gemma2-9b-It",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Basic conversation
response = model.invoke([
    HumanMessage(content="Hi, My name is Krish and I am a Chief AI Engineer")
])
print(response.content)
```

## Memory Management

### In-Memory Chat History

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Create message store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Add memory to model
with_message_history = RunnableWithMessageHistory(
    model,
    get_session_history
)

# Configure session
config = {"configurable": {"session_id": "chat1"}}

# First interaction
response1 = with_message_history.invoke(
    [HumanMessage(content="Hi, My name is Krish and I am a Chief AI Engineer")],
    config=config
)

# Second interaction - should remember context
response2 = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config
)
```

### Multiple Sessions

```python
# Different sessions maintain separate contexts
config1 = {"configurable": {"session_id": "user1"}}
config2 = {"configurable": {"session_id": "user2"}}

# User 1 conversation
response1 = with_message_history.invoke(
    [HumanMessage(content="Hi, I'm John")],
    config=config1
)

# User 2 conversation
response2 = with_message_history.invoke(
    [HumanMessage(content="Hi, I'm Sarah")],
    config=config2
)

# Each session maintains its own context
user1_response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config1
)  # Should respond with "John"

user2_response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config2
)  # Should respond with "Sarah"
```

## Advanced Chatbot with LangGraph

### Stateful Chatbot Implementation

```python
from langgraph.graph import StateGraph, START, END
from typing import List, Dict, Optional

class ChatbotState(TypedDict):
    user_input: str
    chat_history: List[Dict[str, str]]
    user_profile: Dict[str, str]
    current_intent: str
    response: str
    needs_clarification: bool

def understand_intent(state: ChatbotState):
    """Analyze user intent"""
    user_input = state["user_input"].lower()
    
    # Simple intent classification
    if any(word in user_input for word in ["hello", "hi", "hey"]):
        intent = "greeting"
    elif any(word in user_input for word in ["bye", "goodbye", "exit"]):
        intent = "farewell"
    elif any(word in user_input for word in ["help", "support"]):
        intent = "help"
    elif "?" in user_input:
        intent = "question"
    else:
        intent = "general"
    
    return {"current_intent": intent}

def generate_response(state: ChatbotState):
    """Generate appropriate response based on intent"""
    intent = state["current_intent"]
    user_input = state["user_input"]
    
    if intent == "greeting":
        name = state["user_profile"].get("name", "there")
        response = f"Hello {name}! How can I help you today?"
    elif intent == "farewell":
        response = "Goodbye! Have a great day!"
    elif intent == "help":
        response = "I'm here to help! What do you need assistance with?"
    elif intent == "question":
        # Use LLM for complex questions
        model = ChatGroq(model="Gemma2-9b-It", groq_api_key=os.getenv("GROQ_API_KEY"))
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["chat_history"]])
        
        prompt = f"""
        Conversation history:
        {context}
        
        User: {user_input}
        
        Respond helpfully and maintain conversation context.
        """
        
        response = model.invoke([HumanMessage(content=prompt)]).content
    else:
        response = "I understand. Can you tell me more about what you'd like to know?"
    
    return {"response": response}

def update_history(state: ChatbotState):
    """Update conversation history"""
    new_history = state["chat_history"] + [
        {"role": "user", "content": state["user_input"]},
        {"role": "assistant", "content": state["response"]}
    ]
    
    return {"chat_history": new_history}

def extract_user_info(state: ChatbotState):
    """Extract user information from conversation"""
    user_input = state["user_input"]
    user_profile = state["user_profile"].copy()
    
    # Simple name extraction
    if "my name is" in user_input.lower():
        name = user_input.lower().split("my name is")[1].strip().split()[0]
        user_profile["name"] = name.capitalize()
    
    return {"user_profile": user_profile}

# Build chatbot graph
def create_chatbot_graph():
    graph = StateGraph(ChatbotState)
    
    graph.add_node("understand_intent", understand_intent)
    graph.add_node("extract_user_info", extract_user_info)
    graph.add_node("generate_response", generate_response)
    graph.add_node("update_history", update_history)
    
    graph.add_edge(START, "understand_intent")
    graph.add_edge("understand_intent", "extract_user_info")
    graph.add_edge("extract_user_info", "generate_response")
    graph.add_edge("generate_response", "update_history")
    graph.add_edge("update_history", END)
    
    return graph.compile()

# Usage
chatbot = create_chatbot_graph()

# Initialize state
initial_state = {
    "user_input": "",
    "chat_history": [],
    "user_profile": {},
    "current_intent": "",
    "response": "",
    "needs_clarification": False
}

# Conversation loop
def chat_with_bot():
    state = initial_state
    
    print("Chatbot: Hello! I'm here to help. Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
        
        state["user_input"] = user_input
        result = chatbot.invoke(state)
        
        print(f"Chatbot: {result['response']}")
        
        # Update state for next iteration
        state = result

# Run the chatbot
# chat_with_bot()
```

## Web-Based Chatbot

From the `Chatbot_with_Web` project:

### Streamlit Integration

```python
import streamlit as st
from src.langgraphagenticai.main import load_langgraph_agenticai_app
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder

def create_streamlit_chatbot():
    """Create a Streamlit-based chatbot"""
    
    st.title("🤖 AI Chatbot")
    st.write("Welcome to the AI-powered chatbot!")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Initialize LLM
                llm_config = GroqLLM()
                model = llm_config.get_llm_model()
                
                # Create graph
                graph_builder = GraphBuilder(model)
                graph = graph_builder.setup_graph("chatbot")
                
                # Generate response
                response = graph.invoke({"input": prompt})
                
                st.markdown(response["response"])
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["response"]
                })

# Run with: streamlit run chatbot_app.py
```

### FastAPI Backend

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.LLMS.groqllm import GroqLLM

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str

# Store conversations
conversations = {}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Initialize LLM
        llm_config = GroqLLM()
        model = llm_config.get_llm_model()
        
        # Get or create conversation
        if request.session_id not in conversations:
            conversations[request.session_id] = {
                "history": [],
                "graph": GraphBuilder(model).setup_graph("chatbot")
            }
        
        conversation = conversations[request.session_id]
        
        # Process message
        state = {
            "input": request.message,
            "history": conversation["history"]
        }
        
        result = conversation["graph"].invoke(state)
        
        # Update conversation history
        conversation["history"].append({
            "user": request.message,
            "assistant": result["response"]
        })
        
        return ChatResponse(
            response=result["response"],
            session_id=request.session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{session_id}")
async def get_conversation(session_id: str):
    if session_id in conversations:
        return conversations[session_id]["history"]
    return {"error": "Conversation not found"}

# Run with: uvicorn main:app --reload
```

## Chatbot with Tools

From `6-chatbotswithmultipletools.ipynb`:

```python
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

def create_tool_enabled_chatbot():
    """Create a chatbot with access to tools"""
    
    # Define tools
    def search_web(query: str) -> str:
        """Search the web for information"""
        # In a real implementation, use actual web search
        return f"Search results for: {query}"
    
    def calculate(expression: str) -> str:
        """Calculate mathematical expressions"""
        try:
            result = eval(expression)
            return str(result)
        except:
            return "Error in calculation"
    
    def get_weather(location: str) -> str:
        """Get weather information"""
        # In a real implementation, use weather API
        return f"Weather in {location}: Sunny, 25°C"
    
    # Create tools
    tools = [
        Tool(
            name="search",
            func=search_web,
            description="Search the web for information"
        ),
        Tool(
            name="calculator",
            func=calculate,
            description="Calculate mathematical expressions"
        ),
        Tool(
            name="weather",
            func=get_weather,
            description="Get weather information for a location"
        )
    ]
    
    # Create agent
    prompt = PromptTemplate.from_template("""
    You are a helpful assistant with access to the following tools:
    
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
    
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=os.getenv("GROQ_API_KEY"))
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Usage
tool_chatbot = create_tool_enabled_chatbot()

# Examples
response1 = tool_chatbot.invoke({"input": "What's the weather in New York?"})
response2 = tool_chatbot.invoke({"input": "Calculate 25 * 34 + 12"})
response3 = tool_chatbot.invoke({"input": "Search for information about LangChain"})
```

## Context-Aware Chatbot

### Maintaining Context Across Conversations

```python
from datetime import datetime
from typing import Dict, Any

class ContextualChatbot:
    def __init__(self, model):
        self.model = model
        self.user_profiles = {}
        self.conversation_contexts = {}
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get or create user context"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "name": None,
                "preferences": {},
                "conversation_history": [],
                "last_interaction": None
            }
        return self.user_profiles[user_id]
    
    def update_user_profile(self, user_id: str, info: Dict[str, Any]):
        """Update user profile with new information"""
        profile = self.get_user_context(user_id)
        profile.update(info)
        profile["last_interaction"] = datetime.now().isoformat()
    
    def chat(self, user_id: str, message: str) -> str:
        """Process chat message with context"""
        context = self.get_user_context(user_id)
        
        # Build contextual prompt
        prompt_parts = []
        
        # Add user profile context
        if context["name"]:
            prompt_parts.append(f"User name: {context['name']}")
        
        # Add recent conversation history
        if context["conversation_history"]:
            recent_history = context["conversation_history"][-5:]  # Last 5 exchanges
            prompt_parts.append("Recent conversation:")
            for exchange in recent_history:
                prompt_parts.append(f"User: {exchange['user']}")
                prompt_parts.append(f"Assistant: {exchange['assistant']}")
        
        # Add current message
        prompt_parts.append(f"Current message: {message}")
        
        # Generate response
        full_prompt = "\n".join(prompt_parts) + "\n\nRespond helpfully and maintain conversation context."
        
        response = self.model.invoke([HumanMessage(content=full_prompt)]).content
        
        # Update conversation history
        context["conversation_history"].append({
            "user": message,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Extract and update user information
        self._extract_user_info(user_id, message)
        
        return response
    
    def _extract_user_info(self, user_id: str, message: str):
        """Extract user information from message"""
        message_lower = message.lower()
        
        # Extract name
        if "my name is" in message_lower:
            name = message_lower.split("my name is")[1].strip().split()[0]
            self.update_user_profile(user_id, {"name": name.capitalize()})
        
        # Extract preferences
        if "i like" in message_lower or "i prefer" in message_lower:
            # Simple preference extraction
            preference = message_lower.split("like" if "like" in message_lower else "prefer")[1].strip()
            current_prefs = self.get_user_context(user_id)["preferences"]
            current_prefs["general"] = preference
            self.update_user_profile(user_id, {"preferences": current_prefs})

# Usage
contextual_bot = ContextualChatbot(model)

# User conversation
user_id = "user123"
response1 = contextual_bot.chat(user_id, "Hi, my name is Alice")
response2 = contextual_bot.chat(user_id, "I like science fiction books")
response3 = contextual_bot.chat(user_id, "What do you remember about me?")
```

## Conversation Analytics

### Tracking and Analyzing Conversations

```python
from collections import defaultdict
from datetime import datetime
import json

class ConversationAnalytics:
    def __init__(self):
        self.conversation_logs = []
        self.user_metrics = defaultdict(lambda: {
            "total_messages": 0,
            "avg_message_length": 0,
            "common_intents": defaultdict(int),
            "session_count": 0,
            "last_active": None
        })
    
    def log_conversation(self, user_id: str, message: str, response: str, intent: str = None):
        """Log a conversation exchange"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "user_message": message,
            "bot_response": response,
            "intent": intent,
            "message_length": len(message),
            "response_length": len(response)
        }
        
        self.conversation_logs.append(log_entry)
        self._update_user_metrics(user_id, message, intent)
    
    def _update_user_metrics(self, user_id: str, message: str, intent: str):
        """Update user metrics"""
        metrics = self.user_metrics[user_id]
        metrics["total_messages"] += 1
        metrics["avg_message_length"] = (
            (metrics["avg_message_length"] * (metrics["total_messages"] - 1) + len(message)) 
            / metrics["total_messages"]
        )
        
        if intent:
            metrics["common_intents"][intent] += 1
        
        metrics["last_active"] = datetime.now().isoformat()
    
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics for a specific user"""
        return dict(self.user_metrics[user_id])
    
    def get_global_analytics(self) -> Dict[str, Any]:
        """Get global conversation analytics"""
        if not self.conversation_logs:
            return {}
        
        total_conversations = len(self.conversation_logs)
        avg_message_length = sum(log["message_length"] for log in self.conversation_logs) / total_conversations
        avg_response_length = sum(log["response_length"] for log in self.conversation_logs) / total_conversations
        
        intent_distribution = defaultdict(int)
        for log in self.conversation_logs:
            if log["intent"]:
                intent_distribution[log["intent"]] += 1
        
        return {
            "total_conversations": total_conversations,
            "unique_users": len(self.user_metrics),
            "avg_message_length": avg_message_length,
            "avg_response_length": avg_response_length,
            "intent_distribution": dict(intent_distribution),
            "active_users": len([u for u in self.user_metrics.values() if u["last_active"]])
        }
    
    def export_analytics(self, filename: str):
        """Export analytics to JSON file"""
        analytics_data = {
            "global_analytics": self.get_global_analytics(),
            "user_metrics": dict(self.user_metrics),
            "conversation_logs": self.conversation_logs
        }
        
        with open(filename, 'w') as f:
            json.dump(analytics_data, f, indent=2)
```

## Production Considerations

### Error Handling and Fallbacks

```python
class ProductionChatbot:
    def __init__(self, primary_model, fallback_model=None):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.error_count = 0
        self.max_errors = 3
    
    def safe_chat(self, message: str, user_id: str) -> str:
        """Chat with error handling and fallbacks"""
        try:
            # Try primary model
            response = self.primary_model.invoke([HumanMessage(content=message)])
            self.error_count = 0  # Reset error count on success
            return response.content
            
        except Exception as e:
            self.error_count += 1
            print(f"Primary model error: {e}")
            
            if self.fallback_model and self.error_count < self.max_errors:
                try:
                    # Try fallback model
                    response = self.fallback_model.invoke([HumanMessage(content=message)])
                    return response.content
                except Exception as fallback_error:
                    print(f"Fallback model error: {fallback_error}")
            
            # Final fallback - predefined responses
            return self._get_fallback_response(message)
    
    def _get_fallback_response(self, message: str) -> str:
        """Get predefined fallback response"""
        fallback_responses = [
            "I'm sorry, I'm having trouble processing that right now. Can you try rephrasing?",
            "I'm experiencing some technical difficulties. Please try again in a moment.",
            "I apologize, but I'm not able to respond properly at the moment. Is there something else I can help with?"
        ]
        
        # Simple response selection based on message content
        if "?" in message:
            return "I'm sorry, I'm having trouble answering questions right now. Please try again later."
        
        return fallback_responses[self.error_count % len(fallback_responses)]
```

### Rate Limiting and Resource Management

```python
import time
from collections import defaultdict

class RateLimitedChatbot:
    def __init__(self, chatbot, requests_per_minute=10):
        self.chatbot = chatbot
        self.requests_per_minute = requests_per_minute
        self.user_requests = defaultdict(list)
    
    def is_rate_limited(self, user_id: str) -> bool:
        """Check if user is rate limited"""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.user_requests[user_id] = [
            req_time for req_time in self.user_requests[user_id]
            if req_time > minute_ago
        ]
        
        # Check if limit exceeded
        return len(self.user_requests[user_id]) >= self.requests_per_minute
    
    def chat(self, user_id: str, message: str) -> str:
        """Rate-limited chat"""
        if self.is_rate_limited(user_id):
            return "Rate limit exceeded. Please wait a moment before sending another message."
        
        # Record request
        self.user_requests[user_id].append(time.time())
        
        # Process message
        return self.chatbot.safe_chat(message, user_id)
```

## Next Steps

Continue with:

- [08 - Agents and Tools](08-agents-tools.md)
- [09 - Workflows and Orchestration](09-workflows-orchestration.md)
- [10 - Production Deployment](10-production-deployment.md)

## Key Takeaways

- **Memory management** is crucial for meaningful conversations
- **Session handling** enables multiple concurrent users
- **Context awareness** improves conversation quality
- **Tool integration** extends chatbot capabilities
- **Error handling** ensures reliable operation
- **Analytics** help improve chatbot performance
- **Rate limiting** prevents abuse and manages resources
- **Fallback mechanisms** provide graceful degradation

---

**Remember**: Great chatbots combine technical excellence with user experience design. Focus on natural conversation flow, helpful responses, and robust error handling.
