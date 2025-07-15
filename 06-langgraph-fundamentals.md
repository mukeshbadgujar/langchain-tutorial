# 06 - LangGraph Fundamentals

## Overview

LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain's capabilities by providing graph-based workflow orchestration with persistent state management.

## What is LangGraph?

LangGraph enables the creation of complex workflows where:
- Multiple LLM calls can interact
- State persists across interactions
- Conditional logic controls flow
- Parallel processing is possible
- Human intervention can be integrated

### Key Concepts

- **State**: Shared data structure across all nodes
- **Nodes**: Individual processing units (functions)
- **Edges**: Connections between nodes
- **Graph**: Complete workflow structure
- **Compilation**: Process to create executable graph

## Basic Graph Structure

From the project's `1-simplegraph.ipynb`:

### State Definition

```python
from typing_extensions import TypedDict

class State(TypedDict):
    graph_info: str
```

### Node Functions

```python
def start_play(state: State):
    print("Start_Play node has been called")
    return {"graph_info": state['graph_info'] + " I am planning to play"}

def cricket(state: State):
    print("My Cricket node has been called")
    return {"graph_info": state['graph_info'] + " Cricket"}

def badminton(state: State):
    print("My badminton node has been called")
    return {"graph_info": state['graph_info'] + " Badminton"}
```

### Conditional Edge Function

```python
import random
from typing import Literal

def random_play(state: State) -> Literal['cricket', 'badminton']:
    graph_info = state['graph_info']
    
    if random.random() > 0.5:
        return "cricket"
    else:
        return "badminton"
```

### Graph Construction

```python
from langgraph.graph import StateGraph, START, END

# Build Graph
graph = StateGraph(State)

# Adding the nodes
graph.add_node("start_play", start_play)
graph.add_node("cricket", cricket)
graph.add_node("badminton", badminton)

# Schedule the flow of the graph
graph.add_edge(START, "start_play")
graph.add_conditional_edges("start_play", random_play)
graph.add_edge("cricket", END)
graph.add_edge("badminton", END)

# Compile the graph
graph_builder = graph.compile()
```

### Graph Execution

```python
# Execute the graph
result = graph_builder.invoke({"graph_info": "Hey My name is Krish"})
print(result)
```

## Advanced State Management

### Complex State Schema

From `3-DataclassStateSchema.ipynb`:

```python
from dataclasses import dataclass
from typing import Optional, List, Dict
from langgraph.graph import StateGraph

@dataclass
class ComplexState:
    messages: List[str]
    user_info: Dict[str, str]
    current_step: str
    error_count: int
    is_complete: bool
    
    def add_message(self, message: str):
        self.messages.append(message)
    
    def increment_error(self):
        self.error_count += 1
```

### State Reducers

```python
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MessagesState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str
    session_id: str
```

## Pydantic State Models

From `4-pydantic.ipynb`:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class UserState(BaseModel):
    name: str = Field(description="User's name")
    age: Optional[int] = Field(default=None, description="User's age")
    interests: List[str] = Field(default_factory=list, description="User interests")
    conversation_history: List[str] = Field(default_factory=list)
    
    def add_to_history(self, message: str):
        self.conversation_history.append(message)
    
    class Config:
        arbitrary_types_allowed = True

# Usage in graph
def process_user_input(state: UserState):
    # Process user input
    state.add_to_history(f"Processed input for {state.name}")
    return state
```

## Workflow Patterns

### Sequential Processing

```python
from langgraph.graph import StateGraph, START, END

class SequentialState(TypedDict):
    input_data: str
    processed_data: str
    final_result: str

def step_1(state: SequentialState):
    """First processing step"""
    processed = f"Step 1: {state['input_data']}"
    return {"processed_data": processed}

def step_2(state: SequentialState):
    """Second processing step"""
    processed = f"Step 2: {state['processed_data']}"
    return {"processed_data": processed}

def step_3(state: SequentialState):
    """Final processing step"""
    result = f"Final: {state['processed_data']}"
    return {"final_result": result}

# Build sequential graph
graph = StateGraph(SequentialState)
graph.add_node("step_1", step_1)
graph.add_node("step_2", step_2)
graph.add_node("step_3", step_3)

graph.add_edge(START, "step_1")
graph.add_edge("step_1", "step_2")
graph.add_edge("step_2", "step_3")
graph.add_edge("step_3", END)

sequential_graph = graph.compile()
```

### Parallel Processing

From `4-+Workflows/2-parallelization.ipynb`:

```python
from langgraph.graph import StateGraph, START, END

class ParallelState(TypedDict):
    input_text: str
    summary: str
    sentiment: str
    keywords: List[str]
    translation: str

def summarize_text(state: ParallelState):
    """Summarize the input text"""
    # Simulate summarization
    summary = f"Summary of: {state['input_text'][:50]}..."
    return {"summary": summary}

def analyze_sentiment(state: ParallelState):
    """Analyze sentiment of text"""
    # Simulate sentiment analysis
    sentiment = "positive"  # Placeholder
    return {"sentiment": sentiment}

def extract_keywords(state: ParallelState):
    """Extract keywords from text"""
    # Simulate keyword extraction
    keywords = ["AI", "LangChain", "Graph"]
    return {"keywords": keywords}

def translate_text(state: ParallelState):
    """Translate text"""
    # Simulate translation
    translation = f"Translated: {state['input_text']}"
    return {"translation": translation}

def combine_results(state: ParallelState):
    """Combine all parallel results"""
    combined = {
        "summary": state["summary"],
        "sentiment": state["sentiment"],
        "keywords": state["keywords"],
        "translation": state["translation"]
    }
    return combined

# Build parallel graph
graph = StateGraph(ParallelState)

# Add parallel nodes
graph.add_node("summarize", summarize_text)
graph.add_node("sentiment", analyze_sentiment)
graph.add_node("keywords", extract_keywords)
graph.add_node("translate", translate_text)
graph.add_node("combine", combine_results)

# Connect to parallel nodes
graph.add_edge(START, "summarize")
graph.add_edge(START, "sentiment")
graph.add_edge(START, "keywords")
graph.add_edge(START, "translate")

# Combine results
graph.add_edge("summarize", "combine")
graph.add_edge("sentiment", "combine")
graph.add_edge("keywords", "combine")
graph.add_edge("translate", "combine")

graph.add_edge("combine", END)

parallel_graph = graph.compile()
```

### Conditional Routing

From `4-+Workflows/3-Routing.ipynb`:

```python
from typing import Literal

class RoutingState(TypedDict):
    user_input: str
    intent: str
    response: str
    route_taken: str

def classify_intent(state: RoutingState):
    """Classify user intent"""
    user_input = state['user_input'].lower()
    
    if 'weather' in user_input:
        intent = 'weather'
    elif 'news' in user_input:
        intent = 'news'
    elif 'translate' in user_input:
        intent = 'translate'
    else:
        intent = 'general'
    
    return {"intent": intent}

def route_intent(state: RoutingState) -> Literal['weather', 'news', 'translate', 'general']:
    """Route based on classified intent"""
    return state['intent']

def handle_weather(state: RoutingState):
    """Handle weather queries"""
    return {
        "response": "Weather information here",
        "route_taken": "weather"
    }

def handle_news(state: RoutingState):
    """Handle news queries"""
    return {
        "response": "News information here",
        "route_taken": "news"
    }

def handle_translate(state: RoutingState):
    """Handle translation queries"""
    return {
        "response": "Translation result here",
        "route_taken": "translate"
    }

def handle_general(state: RoutingState):
    """Handle general queries"""
    return {
        "response": "General response here",
        "route_taken": "general"
    }

# Build routing graph
graph = StateGraph(RoutingState)

graph.add_node("classify", classify_intent)
graph.add_node("weather", handle_weather)
graph.add_node("news", handle_news)
graph.add_node("translate", handle_translate)
graph.add_node("general", handle_general)

graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route_intent)
graph.add_edge("weather", END)
graph.add_edge("news", END)
graph.add_edge("translate", END)
graph.add_edge("general", END)

routing_graph = graph.compile()
```

## Streaming and Real-time Processing

From `1-streaming.ipynb`:

```python
from langgraph.graph import StateGraph, START, END
import time

class StreamingState(TypedDict):
    messages: List[str]
    current_step: str
    progress: float

def streaming_node_1(state: StreamingState):
    """First streaming node"""
    print("Processing step 1...")
    time.sleep(1)  # Simulate processing
    
    return {
        "messages": state["messages"] + ["Step 1 completed"],
        "current_step": "step_1",
        "progress": 0.33
    }

def streaming_node_2(state: StreamingState):
    """Second streaming node"""
    print("Processing step 2...")
    time.sleep(1)  # Simulate processing
    
    return {
        "messages": state["messages"] + ["Step 2 completed"],
        "current_step": "step_2",
        "progress": 0.66
    }

def streaming_node_3(state: StreamingState):
    """Third streaming node"""
    print("Processing step 3...")
    time.sleep(1)  # Simulate processing
    
    return {
        "messages": state["messages"] + ["Step 3 completed"],
        "current_step": "step_3",
        "progress": 1.0
    }

# Build streaming graph
graph = StateGraph(StreamingState)

graph.add_node("step_1", streaming_node_1)
graph.add_node("step_2", streaming_node_2)
graph.add_node("step_3", streaming_node_3)

graph.add_edge(START, "step_1")
graph.add_edge("step_1", "step_2")
graph.add_edge("step_2", "step_3")
graph.add_edge("step_3", END)

streaming_graph = graph.compile()

# Stream execution
for step in streaming_graph.stream({"messages": [], "current_step": "", "progress": 0.0}):
    print(f"Current state: {step}")
```

## Human-in-the-Loop

From `5-HumanintheLoop/1-Humanintheloop.ipynb`:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolInvocation
from typing import List

class HumanLoopState(TypedDict):
    messages: List[str]
    human_input: str
    needs_approval: bool
    approved: bool

def process_request(state: HumanLoopState):
    """Process initial request"""
    return {
        "messages": state["messages"] + ["Request processed"],
        "needs_approval": True
    }

def request_human_approval(state: HumanLoopState):
    """Request human approval"""
    print("Requesting human approval...")
    print(f"Current state: {state}")
    
    # In a real application, this would wait for user input
    # For demo purposes, we'll simulate approval
    human_input = input("Approve? (y/n): ")
    
    return {
        "human_input": human_input,
        "approved": human_input.lower() == 'y'
    }

def handle_approval(state: HumanLoopState):
    """Handle approval result"""
    if state["approved"]:
        return {
            "messages": state["messages"] + ["Approved and processed"],
            "needs_approval": False
        }
    else:
        return {
            "messages": state["messages"] + ["Rejected"],
            "needs_approval": False
        }

def check_approval_needed(state: HumanLoopState) -> str:
    """Check if approval is needed"""
    if state["needs_approval"]:
        return "request_approval"
    else:
        return "complete"

def complete_process(state: HumanLoopState):
    """Complete the process"""
    return {
        "messages": state["messages"] + ["Process completed"]
    }

# Build human-in-the-loop graph
graph = StateGraph(HumanLoopState)

graph.add_node("process", process_request)
graph.add_node("request_approval", request_human_approval)
graph.add_node("handle_approval", handle_approval)
graph.add_node("complete", complete_process)

graph.add_edge(START, "process")
graph.add_conditional_edges("process", check_approval_needed)
graph.add_edge("request_approval", "handle_approval")
graph.add_edge("handle_approval", "complete")
graph.add_edge("complete", END)

human_loop_graph = graph.compile()
```

## Agent Integration

From `7-ReActAgents.ipynb`:

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    input: str
    agent_outcome: str
    intermediate_steps: List[tuple]

def create_agent_node(tools, llm):
    """Create an agent node"""
    
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
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    def agent_node(state: AgentState):
        """Execute agent"""
        result = agent_executor.invoke({"input": state["input"]})
        return {"agent_outcome": result["output"]}
    
    return agent_node

# Example tools
def search_tool(query: str) -> str:
    """Search for information"""
    return f"Search results for: {query}"

def calculator_tool(expression: str) -> str:
    """Calculate mathematical expressions"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Error in calculation"

tools = [
    Tool(name="search", func=search_tool, description="Search for information"),
    Tool(name="calculator", func=calculator_tool, description="Calculate expressions")
]

# Build agent graph
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
agent_node = create_agent_node(tools, llm)

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

agent_graph = graph.compile()
```

## Error Handling and Recovery

```python
from typing import Optional

class ErrorHandlingState(TypedDict):
    input_data: str
    result: Optional[str]
    error_count: int
    max_retries: int
    last_error: Optional[str]

def risky_operation(state: ErrorHandlingState):
    """Operation that might fail"""
    try:
        # Simulate risky operation
        import random
        if random.random() < 0.3:  # 30% chance of failure
            raise Exception("Random failure occurred")
        
        return {
            "result": f"Success: {state['input_data']}",
            "error_count": 0
        }
    except Exception as e:
        return {
            "error_count": state["error_count"] + 1,
            "last_error": str(e)
        }

def check_retry_needed(state: ErrorHandlingState) -> str:
    """Check if retry is needed"""
    if state.get("result"):
        return "success"
    elif state["error_count"] >= state["max_retries"]:
        return "failure"
    else:
        return "retry"

def handle_success(state: ErrorHandlingState):
    """Handle successful operation"""
    return {"result": f"Final result: {state['result']}"}

def handle_failure(state: ErrorHandlingState):
    """Handle operation failure"""
    return {
        "result": f"Failed after {state['error_count']} attempts. Last error: {state['last_error']}"
    }

def retry_operation(state: ErrorHandlingState):
    """Retry the operation"""
    print(f"Retrying operation (attempt {state['error_count'] + 1})")
    return risky_operation(state)

# Build error handling graph
graph = StateGraph(ErrorHandlingState)

graph.add_node("operation", risky_operation)
graph.add_node("retry", retry_operation)
graph.add_node("success", handle_success)
graph.add_node("failure", handle_failure)

graph.add_edge(START, "operation")
graph.add_conditional_edges("operation", check_retry_needed)
graph.add_conditional_edges("retry", check_retry_needed)
graph.add_edge("success", END)
graph.add_edge("failure", END)

error_handling_graph = graph.compile()

# Usage
result = error_handling_graph.invoke({
    "input_data": "test data",
    "result": None,
    "error_count": 0,
    "max_retries": 3,
    "last_error": None
})
```

## Graph Visualization

```python
from IPython.display import Image, display

def visualize_graph(graph):
    """Visualize the graph structure"""
    try:
        # Generate Mermaid diagram
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception as e:
        print(f"Visualization error: {e}")
        # Fallback to text representation
        print("Graph nodes:", graph.get_graph().nodes)
        print("Graph edges:", graph.get_graph().edges)

# Usage
visualize_graph(graph_builder)
```

## Persistence and Checkpointing

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

# Create graph with persistence
memory = MemorySaver()

graph = StateGraph(State)
# ... add nodes and edges ...

# Compile with checkpointing
persistent_graph = graph.compile(checkpointer=memory)

# Use with thread ID for persistence
config = {"configurable": {"thread_id": "conversation_1"}}
result = persistent_graph.invoke(initial_state, config=config)

# Continue conversation
result2 = persistent_graph.invoke(next_input, config=config)
```

## Best Practices

### 1. State Design

```python
# Good: Clear, typed state
class WellDesignedState(TypedDict):
    user_id: str
    conversation_history: List[str]
    current_intent: str
    confidence_score: float
    last_updated: str

# Avoid: Unclear, untyped state
class PoorState(TypedDict):
    data: dict  # Too generic
    stuff: str  # Unclear purpose
```

### 2. Node Design

```python
# Good: Single responsibility
def validate_input(state: State):
    """Validate user input"""
    # Only handle input validation
    pass

def process_query(state: State):
    """Process the validated query"""
    # Only handle query processing
    pass

# Avoid: Multiple responsibilities
def validate_and_process(state: State):
    """Validate and process - doing too much"""
    # Handle both validation and processing
    pass
```

### 3. Error Handling

```python
def robust_node(state: State):
    """Node with proper error handling"""
    try:
        # Main logic
        result = process_data(state)
        return {"result": result}
    except SpecificError as e:
        # Handle specific error
        return {"error": f"Specific error: {e}"}
    except Exception as e:
        # Handle unexpected errors
        return {"error": f"Unexpected error: {e}"}
```

### 4. Testing

```python
import unittest

class TestGraph(unittest.TestCase):
    def setUp(self):
        self.graph = create_test_graph()
    
    def test_happy_path(self):
        """Test normal execution"""
        result = self.graph.invoke({"input": "test"})
        self.assertIn("result", result)
    
    def test_error_handling(self):
        """Test error conditions"""
        result = self.graph.invoke({"input": "invalid"})
        self.assertIn("error", result)
```

## Next Steps

Continue with:

- [07 - Chatbots and Conversational AI](07-chatbots-conversational-ai.md)
- [08 - Agents and Tools](08-agents-tools.md)
- [09 - Workflows and Orchestration](09-workflows-orchestration.md)

## Key Takeaways

- **LangGraph** enables stateful, multi-step LLM workflows
- **State management** is central to graph design
- **Nodes** are functions that process and transform state
- **Edges** define the flow between nodes
- **Conditional routing** enables dynamic workflow paths
- **Parallel processing** improves performance
- **Error handling** ensures robust execution
- **Persistence** enables long-running conversations
- **Visualization** helps understand graph structure

---

**Remember**: Design your graphs with clear state schemas, single-responsibility nodes, and proper error handling. Start simple and add complexity gradually.
