# 08 - Agents and Tools

## Overview

Agents are autonomous systems that can reason, plan, and execute actions using tools. This document covers ReAct agents, tool integration, and multi-agent workflows based on the project implementations.

## Understanding Agents

### What Are Agents?

Agents are AI systems that can:
- **Reason**: Analyze problems and plan solutions
- **Act**: Execute actions using tools
- **Observe**: Process results and adapt behavior
- **Iterate**: Repeat the process until completion

### Agent Architecture

```
Input → Agent → Tool Selection → Tool Execution → Observation → Decision → Output
   ↓        ↓           ↓              ↓             ↓          ↓        ↓
Question  LLM    Choose Tool    Execute Action   Get Result  Continue  Answer
```

## ReAct Agents

From `7-ReActAgents.ipynb`, ReAct (Reasoning + Acting) agents interleave reasoning and acting.

### Basic ReAct Agent

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_groq import ChatGroq
import os

def create_basic_react_agent():
    """Create a basic ReAct agent"""
    
    # Define tools
    def search_tool(query: str) -> str:
        """Search for information"""
        # In production, use actual search API
        return f"Search results for '{query}': Information found about {query}"
    
    def calculator_tool(expression: str) -> str:
        """Calculate mathematical expressions"""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except:
            return "Error: Invalid mathematical expression"
    
    def weather_tool(location: str) -> str:
        """Get weather information"""
        # In production, use actual weather API
        return f"Weather in {location}: Sunny, 22°C"
    
    # Create tool objects
    tools = [
        Tool(
            name="search",
            func=search_tool,
            description="Search for information on the internet"
        ),
        Tool(
            name="calculator",
            func=calculator_tool,
            description="Calculate mathematical expressions"
        ),
        Tool(
            name="weather",
            func=weather_tool,
            description="Get current weather for a location"
        )
    ]
    
    # Create ReAct prompt
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
    
    # Initialize LLM
    llm = ChatGroq(
        model="Gemma2-9b-It",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

# Usage
react_agent = create_basic_react_agent()

# Examples
response1 = react_agent.invoke({"input": "What's the weather in New York?"})
response2 = react_agent.invoke({"input": "Calculate 15 * 23 + 45"})
response3 = react_agent.invoke({"input": "Search for information about LangChain"})
```

## Tool Integration

### Custom Tools

```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to calculate")

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Useful for calculating mathematical expressions"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        """Execute the tool"""
        try:
            result = eval(expression)
            return f"The result is: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        """Async version of the tool"""
        return self._run(expression)

# File System Tool
class FileSystemInput(BaseModel):
    path: str = Field(description="File path to read")

class FileReaderTool(BaseTool):
    name = "file_reader"
    description = "Read contents of a file"
    args_schema: Type[BaseModel] = FileSystemInput
    
    def _run(self, path: str) -> str:
        """Read file contents"""
        try:
            with open(path, 'r') as f:
                content = f.read()
            return f"File contents:\n{content}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    async def _arun(self, path: str) -> str:
        return self._run(path)

# Usage
custom_tools = [CalculatorTool(), FileReaderTool()]
```

### Web Search Tool

```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

def create_web_search_tool():
    """Create a web search tool"""
    
    # Initialize search wrapper
    search_wrapper = DuckDuckGoSearchAPIWrapper(
        max_results=5,
        region="us-en"
    )
    
    # Create search tool
    search_tool = DuckDuckGoSearchRun(
        api_wrapper=search_wrapper,
        name="web_search",
        description="Search the web for current information"
    )
    
    return search_tool

# Wikipedia Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

def create_wikipedia_tool():
    """Create a Wikipedia search tool"""
    
    wikipedia_wrapper = WikipediaAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=2000
    )
    
    wikipedia_tool = WikipediaQueryRun(
        api_wrapper=wikipedia_wrapper,
        name="wikipedia",
        description="Search Wikipedia for information"
    )
    
    return wikipedia_tool
```

### Database Tool

```python
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase

class DatabaseInput(BaseModel):
    query: str = Field(description="SQL query to execute")

class DatabaseTool(BaseTool):
    name = "database_query"
    description = "Execute SQL queries on the database"
    args_schema: Type[BaseModel] = DatabaseInput
    
    def __init__(self, db_connection_string: str):
        super().__init__()
        self.db = SQLDatabase.from_uri(db_connection_string)
    
    def _run(self, query: str) -> str:
        """Execute SQL query"""
        try:
            result = self.db.run(query)
            return f"Query result:\n{result}"
        except Exception as e:
            return f"Database error: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

# Usage
# db_tool = DatabaseTool("sqlite:///example.db")
```

## Multi-Agent Workflows

### Agent Communication

```python
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class AgentType(Enum):
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    WRITER = "writer"
    REVIEWER = "reviewer"

@dataclass
class Message:
    sender: str
    recipient: str
    content: str
    message_type: str
    timestamp: str

class MultiAgentSystem:
    def __init__(self):
        self.agents = {}
        self.message_queue = []
        self.shared_memory = {}
    
    def register_agent(self, agent_id: str, agent_type: AgentType, agent_instance):
        """Register an agent in the system"""
        self.agents[agent_id] = {
            "type": agent_type,
            "instance": agent_instance,
            "status": "idle"
        }
    
    def send_message(self, sender: str, recipient: str, content: str, message_type: str = "task"):
        """Send message between agents"""
        message = Message(
            sender=sender,
            recipient=recipient,
            content=content,
            message_type=message_type,
            timestamp=datetime.now().isoformat()
        )
        self.message_queue.append(message)
    
    def process_messages(self):
        """Process all messages in the queue"""
        while self.message_queue:
            message = self.message_queue.pop(0)
            
            if message.recipient in self.agents:
                agent = self.agents[message.recipient]
                self._handle_message(agent, message)
    
    def _handle_message(self, agent: Dict[str, Any], message: Message):
        """Handle message processing for an agent"""
        agent_instance = agent["instance"]
        
        if message.message_type == "task":
            # Process task message
            result = agent_instance.invoke({"input": message.content})
            
            # Store result in shared memory
            self.shared_memory[f"{message.sender}_{message.recipient}"] = result
            
            # Send completion message back
            self.send_message(
                sender=message.recipient,
                recipient=message.sender,
                content=f"Task completed: {result}",
                message_type="completion"
            )

# Example: Research and Writing Workflow
def create_research_writing_workflow():
    """Create a multi-agent research and writing workflow"""
    
    # Create specialized agents
    researcher_tools = [create_web_search_tool(), create_wikipedia_tool()]
    research_agent = create_react_agent_with_tools(researcher_tools, "research")
    
    writer_tools = [CalculatorTool()]  # Writing-specific tools
    writer_agent = create_react_agent_with_tools(writer_tools, "writing")
    
    # Initialize multi-agent system
    mas = MultiAgentSystem()
    mas.register_agent("researcher", AgentType.RESEARCHER, research_agent)
    mas.register_agent("writer", AgentType.WRITER, writer_agent)
    
    # Workflow execution
    def execute_research_workflow(topic: str):
        # Step 1: Research
        mas.send_message(
            sender="coordinator",
            recipient="researcher",
            content=f"Research information about: {topic}",
            message_type="task"
        )
        
        mas.process_messages()
        
        # Step 2: Writing
        research_results = mas.shared_memory.get("coordinator_researcher", "")
        
        mas.send_message(
            sender="coordinator",
            recipient="writer",
            content=f"Write an article based on this research: {research_results}",
            message_type="task"
        )
        
        mas.process_messages()
        
        # Get final result
        final_article = mas.shared_memory.get("coordinator_writer", "")
        return final_article
    
    return execute_research_workflow

# Usage
workflow = create_research_writing_workflow()
result = workflow("LangChain applications in 2024")
```

## Retrieval-Augmented Agents

From the project's agentic RAG implementation:

### RAG Agent with Multiple Knowledge Bases

```python
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def create_rag_agent():
    """Create an agent with multiple retrieval tools"""
    
    # LangGraph Knowledge Base
    langgraph_urls = [
        "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
        "https://langchain-ai.github.io/langgraph/tutorials/workflows/"
    ]
    
    langgraph_docs = [WebBaseLoader(url).load() for url in langgraph_urls]
    langgraph_docs_flat = [item for sublist in langgraph_docs for item in sublist]
    
    # Create vector store
    langgraph_vectorstore = FAISS.from_documents(
        langgraph_docs_flat, 
        OpenAIEmbeddings()
    )
    langgraph_retriever = langgraph_vectorstore.as_retriever()
    
    # LangChain Knowledge Base
    langchain_urls = [
        "https://python.langchain.com/docs/tutorials/",
        "https://python.langchain.com/docs/tutorials/chatbot/"
    ]
    
    langchain_docs = [WebBaseLoader(url).load() for url in langchain_urls]
    langchain_docs_flat = [item for sublist in langchain_docs for item in sublist]
    
    langchain_vectorstore = FAISS.from_documents(
        langchain_docs_flat, 
        OpenAIEmbeddings()
    )
    langchain_retriever = langchain_vectorstore.as_retriever()
    
    # Create retriever tools
    langgraph_tool = create_retriever_tool(
        langgraph_retriever,
        "langgraph_docs",
        "Search LangGraph documentation for workflows, state management, and graph construction"
    )
    
    langchain_tool = create_retriever_tool(
        langchain_retriever,
        "langchain_docs",
        "Search LangChain documentation for basic concepts, chains, and integrations"
    )
    
    # Additional tools
    web_search_tool = create_web_search_tool()
    calculator_tool = CalculatorTool()
    
    # Combine all tools
    all_tools = [langgraph_tool, langchain_tool, web_search_tool, calculator_tool]
    
    # Create agent
    prompt = PromptTemplate.from_template("""
    You are an AI assistant specialized in LangChain and LangGraph. You have access to documentation, 
    web search, and calculation tools. Use the most appropriate tool for each question.

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
    agent = create_react_agent(llm, all_tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        handle_parsing_errors=True
    )

# Usage
rag_agent = create_rag_agent()

# Example queries
response1 = rag_agent.invoke({
    "input": "What is the difference between LangChain and LangGraph?"
})

response2 = rag_agent.invoke({
    "input": "How do I create a simple workflow in LangGraph?"
})
```

## Human-in-the-Loop Agents

From `5-HumanintheLoop`:

### Interactive Agent System

```python
from typing import Optional
import json

class HumanInTheLoopAgent:
    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.human_approval_required = True
        self.pending_actions = []
    
    def set_human_approval(self, required: bool):
        """Enable/disable human approval requirement"""
        self.human_approval_required = required
    
    def execute_with_approval(self, input_data: str) -> str:
        """Execute agent with human approval checkpoints"""
        
        if not self.human_approval_required:
            return self.base_agent.invoke({"input": input_data})
        
        # Step 1: Plan actions
        plan = self._create_plan(input_data)
        
        # Step 2: Get human approval for plan
        approved_plan = self._get_human_approval(plan)
        
        if not approved_plan:
            return "Task cancelled by human operator"
        
        # Step 3: Execute approved plan
        return self._execute_plan(approved_plan)
    
    def _create_plan(self, input_data: str) -> Dict[str, Any]:
        """Create execution plan"""
        # Simulate plan creation
        return {
            "input": input_data,
            "planned_actions": [
                "Search for information",
                "Analyze results",
                "Generate response"
            ],
            "estimated_time": "30 seconds",
            "risk_level": "low"
        }
    
    def _get_human_approval(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get human approval for plan"""
        print("\n" + "="*50)
        print("HUMAN APPROVAL REQUIRED")
        print("="*50)
        print(f"Input: {plan['input']}")
        print(f"Planned Actions: {plan['planned_actions']}")
        print(f"Estimated Time: {plan['estimated_time']}")
        print(f"Risk Level: {plan['risk_level']}")
        print("="*50)
        
        while True:
            approval = input("Approve this plan? (y/n/modify): ").strip().lower()
            
            if approval == 'y':
                return plan
            elif approval == 'n':
                return None
            elif approval == 'modify':
                # Allow human to modify plan
                modified_plan = self._modify_plan(plan)
                return modified_plan
            else:
                print("Please enter 'y' for yes, 'n' for no, or 'modify' to modify the plan")
    
    def _modify_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Allow human to modify the plan"""
        print("\nCurrent plan:")
        print(json.dumps(plan, indent=2))
        
        # Simple modification interface
        new_input = input(f"Modify input (current: {plan['input']}): ").strip()
        if new_input:
            plan['input'] = new_input
        
        return plan
    
    def _execute_plan(self, plan: Dict[str, Any]) -> str:
        """Execute the approved plan"""
        print("\nExecuting approved plan...")
        
        # Execute with base agent
        result = self.base_agent.invoke({"input": plan['input']})
        
        # Final human review
        print("\n" + "="*50)
        print("EXECUTION COMPLETE - HUMAN REVIEW")
        print("="*50)
        print(f"Result: {result}")
        print("="*50)
        
        review = input("Accept this result? (y/n): ").strip().lower()
        
        if review == 'y':
            return result
        else:
            return "Result rejected by human operator"

# Usage
base_agent = create_basic_react_agent()
human_agent = HumanInTheLoopAgent(base_agent)

# Execute with human approval
result = human_agent.execute_with_approval("What is the weather in New York?")
```

## Agent Monitoring and Debugging

### Agent Performance Tracking

```python
from datetime import datetime
from typing import List, Dict, Any
import time

class AgentMonitor:
    def __init__(self):
        self.execution_logs = []
        self.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0,
            "tool_usage_stats": {}
        }
    
    def log_execution(self, agent_id: str, input_data: str, output: str, 
                     execution_time: float, tools_used: List[str], success: bool):
        """Log agent execution"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "input": input_data,
            "output": output,
            "execution_time": execution_time,
            "tools_used": tools_used,
            "success": success
        }
        
        self.execution_logs.append(log_entry)
        self._update_metrics(execution_time, tools_used, success)
    
    def _update_metrics(self, execution_time: float, tools_used: List[str], success: bool):
        """Update performance metrics"""
        self.performance_metrics["total_executions"] += 1
        
        if success:
            self.performance_metrics["successful_executions"] += 1
        else:
            self.performance_metrics["failed_executions"] += 1
        
        # Update average execution time
        total_time = (self.performance_metrics["average_execution_time"] * 
                     (self.performance_metrics["total_executions"] - 1) + execution_time)
        self.performance_metrics["average_execution_time"] = total_time / self.performance_metrics["total_executions"]
        
        # Update tool usage stats
        for tool in tools_used:
            if tool not in self.performance_metrics["tool_usage_stats"]:
                self.performance_metrics["tool_usage_stats"][tool] = 0
            self.performance_metrics["tool_usage_stats"][tool] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        success_rate = (self.performance_metrics["successful_executions"] / 
                       self.performance_metrics["total_executions"]) * 100
        
        return {
            "total_executions": self.performance_metrics["total_executions"],
            "success_rate": f"{success_rate:.2f}%",
            "average_execution_time": f"{self.performance_metrics['average_execution_time']:.2f}s",
            "tool_usage_stats": self.performance_metrics["tool_usage_stats"],
            "recent_executions": self.execution_logs[-5:] if len(self.execution_logs) > 5 else self.execution_logs
        }

# Monitored Agent Wrapper
class MonitoredAgent:
    def __init__(self, agent, monitor: AgentMonitor, agent_id: str):
        self.agent = agent
        self.monitor = monitor
        self.agent_id = agent_id
    
    def invoke(self, input_data: Dict[str, Any]) -> str:
        """Invoke agent with monitoring"""
        start_time = time.time()
        tools_used = []
        success = False
        output = ""
        
        try:
            # Execute agent
            result = self.agent.invoke(input_data)
            output = result if isinstance(result, str) else str(result)
            
            # Extract tools used (this would need to be implemented based on agent internals)
            tools_used = self._extract_tools_used(result)
            success = True
            
        except Exception as e:
            output = f"Error: {str(e)}"
            success = False
        
        execution_time = time.time() - start_time
        
        # Log execution
        self.monitor.log_execution(
            agent_id=self.agent_id,
            input_data=str(input_data),
            output=output,
            execution_time=execution_time,
            tools_used=tools_used,
            success=success
        )
        
        return output
    
    def _extract_tools_used(self, result) -> List[str]:
        """Extract tools used from result"""
        # This would need to be implemented based on your agent's structure
        # For now, return empty list
        return []

# Usage
monitor = AgentMonitor()
base_agent = create_basic_react_agent()
monitored_agent = MonitoredAgent(base_agent, monitor, "react_agent_1")

# Execute with monitoring
result = monitored_agent.invoke({"input": "What is the weather in Paris?"})

# Get performance report
report = monitor.get_performance_report()
print(json.dumps(report, indent=2))
```

## Advanced Agent Patterns

### Hierarchical Agents

```python
class HierarchicalAgentSystem:
    def __init__(self):
        self.coordinator = None
        self.specialist_agents = {}
        self.task_queue = []
    
    def set_coordinator(self, agent):
        """Set the coordinator agent"""
        self.coordinator = agent
    
    def add_specialist(self, name: str, agent, specialization: str):
        """Add specialist agent"""
        self.specialist_agents[name] = {
            "agent": agent,
            "specialization": specialization,
            "busy": False
        }
    
    def delegate_task(self, task: str) -> str:
        """Delegate task through hierarchy"""
        # Coordinator decides task distribution
        delegation_prompt = f"""
        Task: {task}
        
        Available specialists:
        {self._get_specialist_info()}
        
        Which specialist should handle this task? Respond with just the specialist name.
        """
        
        selected_specialist = self.coordinator.invoke({"input": delegation_prompt})
        
        # Execute with selected specialist
        if selected_specialist in self.specialist_agents:
            specialist = self.specialist_agents[selected_specialist]
            if not specialist["busy"]:
                specialist["busy"] = True
                result = specialist["agent"].invoke({"input": task})
                specialist["busy"] = False
                return result
        
        # Fallback to coordinator
        return self.coordinator.invoke({"input": task})
    
    def _get_specialist_info(self) -> str:
        """Get specialist information"""
        info = []
        for name, details in self.specialist_agents.items():
            status = "busy" if details["busy"] else "available"
            info.append(f"- {name}: {details['specialization']} ({status})")
        return "\n".join(info)
```

## Next Steps

Continue with:

- [09 - Workflows and Orchestration](09-workflows-orchestration.md)
- [10 - Production Deployment](10-production-deployment.md)
- [00 - Table of Contents](00-table-of-contents.md)

## Key Takeaways

- **Agents** combine reasoning and action for autonomous problem-solving
- **ReAct pattern** interleaves thinking and acting for better performance
- **Tools** extend agent capabilities with external functions
- **Multi-agent systems** enable complex workflows through collaboration
- **Human-in-the-loop** adds oversight and control to agent actions
- **Monitoring** is essential for production agent systems
- **Hierarchical structures** enable scalable agent organizations
- **Error handling** ensures robust agent operation

---

**Remember**: Agents are powerful but require careful design, monitoring, and safety measures. Start with simple single-agent systems before building complex multi-agent workflows.
