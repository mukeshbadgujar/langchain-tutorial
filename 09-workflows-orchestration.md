# 09 - Workflows and Orchestration

## Overview

This document covers advanced workflow patterns, orchestration techniques, and production-ready implementations using LangGraph and LangChain, based on the project's workflow implementations.

## Workflow Design Patterns

### 1. Sequential Processing

Linear workflows where each step depends on the previous one.

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import List, Dict, Any

class SequentialState(TypedDict):
    input_data: str
    step1_result: str
    step2_result: str
    step3_result: str
    final_output: str
    metadata: Dict[str, Any]

def data_ingestion(state: SequentialState) -> SequentialState:
    """Step 1: Ingest and validate data"""
    data = state["input_data"]
    
    # Simulate data validation
    if not data or len(data) < 10:
        raise ValueError("Insufficient data provided")
    
    processed_data = data.strip().upper()
    
    return {
        "step1_result": processed_data,
        "metadata": {
            "step1_completed": True,
            "data_length": len(processed_data)
        }
    }

def data_processing(state: SequentialState) -> SequentialState:
    """Step 2: Process the data"""
    input_data = state["step1_result"]
    
    # Simulate complex processing
    processed = f"PROCESSED: {input_data}"
    
    return {
        "step2_result": processed,
        "metadata": {
            **state.get("metadata", {}),
            "step2_completed": True,
            "processing_type": "standard"
        }
    }

def data_analysis(state: SequentialState) -> SequentialState:
    """Step 3: Analyze the processed data"""
    processed_data = state["step2_result"]
    
    # Simulate analysis
    analysis = f"ANALYSIS: {processed_data} - Word count: {len(processed_data.split())}"
    
    return {
        "step3_result": analysis,
        "metadata": {
            **state.get("metadata", {}),
            "step3_completed": True,
            "analysis_type": "word_count"
        }
    }

def generate_report(state: SequentialState) -> SequentialState:
    """Step 4: Generate final report"""
    analysis = state["step3_result"]
    metadata = state.get("metadata", {})
    
    report = f"""
    WORKFLOW REPORT
    ===============
    Original Input: {state['input_data']}
    
    Processing Steps:
    1. Data Ingestion: {metadata.get('step1_completed', False)}
    2. Data Processing: {metadata.get('step2_completed', False)}
    3. Data Analysis: {metadata.get('step3_completed', False)}
    
    Final Analysis: {analysis}
    
    Metadata: {metadata}
    """
    
    return {"final_output": report}

# Create sequential workflow
def create_sequential_workflow():
    graph = StateGraph(SequentialState)
    
    # Add nodes in sequence
    graph.add_node("data_ingestion", data_ingestion)
    graph.add_node("data_processing", data_processing)
    graph.add_node("data_analysis", data_analysis)
    graph.add_node("generate_report", generate_report)
    
    # Add sequential edges
    graph.add_edge(START, "data_ingestion")
    graph.add_edge("data_ingestion", "data_processing")
    graph.add_edge("data_processing", "data_analysis")
    graph.add_edge("data_analysis", "generate_report")
    graph.add_edge("generate_report", END)
    
    return graph.compile()

# Usage
sequential_workflow = create_sequential_workflow()
result = sequential_workflow.invoke({
    "input_data": "This is sample data for processing"
})
```

### 2. Parallel Processing

Multiple independent tasks executed simultaneously.

```python
from concurrent.futures import ThreadPoolExecutor
import time

class ParallelState(TypedDict):
    input_text: str
    sentiment_analysis: str
    keyword_extraction: str
    text_summarization: str
    language_detection: str
    combined_results: Dict[str, str]

def analyze_sentiment(state: ParallelState) -> ParallelState:
    """Analyze sentiment of the text"""
    text = state["input_text"]
    
    # Simulate sentiment analysis
    time.sleep(1)  # Simulate processing time
    
    if "good" in text.lower() or "great" in text.lower():
        sentiment = "Positive"
    elif "bad" in text.lower() or "terrible" in text.lower():
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return {"sentiment_analysis": f"Sentiment: {sentiment}"}

def extract_keywords(state: ParallelState) -> ParallelState:
    """Extract keywords from the text"""
    text = state["input_text"]
    
    # Simulate keyword extraction
    time.sleep(1.5)  # Simulate processing time
    
    words = text.split()
    keywords = [word for word in words if len(word) > 4][:3]
    
    return {"keyword_extraction": f"Keywords: {', '.join(keywords)}"}

def summarize_text(state: ParallelState) -> ParallelState:
    """Summarize the text"""
    text = state["input_text"]
    
    # Simulate text summarization
    time.sleep(2)  # Simulate processing time
    
    summary = f"Summary: {text[:50]}..." if len(text) > 50 else f"Summary: {text}"
    
    return {"text_summarization": summary}

def detect_language(state: ParallelState) -> ParallelState:
    """Detect language of the text"""
    text = state["input_text"]
    
    # Simulate language detection
    time.sleep(0.5)  # Simulate processing time
    
    # Simple heuristic
    if any(word in text.lower() for word in ["the", "and", "or", "but"]):
        language = "English"
    else:
        language = "Unknown"
    
    return {"language_detection": f"Language: {language}"}

def combine_parallel_results(state: ParallelState) -> ParallelState:
    """Combine results from parallel processing"""
    results = {
        "sentiment": state.get("sentiment_analysis", "Not available"),
        "keywords": state.get("keyword_extraction", "Not available"),
        "summary": state.get("text_summarization", "Not available"),
        "language": state.get("language_detection", "Not available")
    }
    
    combined_report = f"""
    PARALLEL PROCESSING RESULTS
    ===========================
    Input: {state['input_text']}
    
    {results['sentiment']}
    {results['keywords']}
    {results['summary']}
    {results['language']}
    
    Processing completed in parallel.
    """
    
    return {"combined_results": results}

# Create parallel workflow
def create_parallel_workflow():
    graph = StateGraph(ParallelState)
    
    # Add parallel processing nodes
    graph.add_node("analyze_sentiment", analyze_sentiment)
    graph.add_node("extract_keywords", extract_keywords)
    graph.add_node("summarize_text", summarize_text)
    graph.add_node("detect_language", detect_language)
    graph.add_node("combine_results", combine_parallel_results)
    
    # All parallel nodes start from START
    graph.add_edge(START, "analyze_sentiment")
    graph.add_edge(START, "extract_keywords")
    graph.add_edge(START, "summarize_text")
    graph.add_edge(START, "detect_language")
    
    # All parallel nodes feed into combine_results
    graph.add_edge("analyze_sentiment", "combine_results")
    graph.add_edge("extract_keywords", "combine_results")
    graph.add_edge("summarize_text", "combine_results")
    graph.add_edge("detect_language", "combine_results")
    
    graph.add_edge("combine_results", END)
    
    return graph.compile()

# Usage
parallel_workflow = create_parallel_workflow()
start_time = time.time()
result = parallel_workflow.invoke({
    "input_text": "This is a great example of parallel processing in action"
})
end_time = time.time()
print(f"Parallel processing completed in {end_time - start_time:.2f} seconds")
```

### 3. Conditional Routing

Dynamic workflow paths based on conditions.

```python
from typing import Literal

class RoutingState(TypedDict):
    user_input: str
    user_type: str
    processing_path: str
    result: str
    requires_approval: bool

def classify_user(state: RoutingState) -> RoutingState:
    """Classify user type"""
    user_input = state["user_input"].lower()
    
    if "admin" in user_input or "administrator" in user_input:
        user_type = "admin"
    elif "premium" in user_input or "pro" in user_input:
        user_type = "premium"
    else:
        user_type = "standard"
    
    return {
        "user_type": user_type,
        "requires_approval": user_type == "admin"
    }

def admin_processing(state: RoutingState) -> RoutingState:
    """Process admin requests"""
    result = f"Admin processing: {state['user_input']} - Full access granted"
    return {
        "result": result,
        "processing_path": "admin"
    }

def premium_processing(state: RoutingState) -> RoutingState:
    """Process premium user requests"""
    result = f"Premium processing: {state['user_input']} - Enhanced features available"
    return {
        "result": result,
        "processing_path": "premium"
    }

def standard_processing(state: RoutingState) -> RoutingState:
    """Process standard user requests"""
    result = f"Standard processing: {state['user_input']} - Basic features available"
    return {
        "result": result,
        "processing_path": "standard"
    }

def approval_check(state: RoutingState) -> RoutingState:
    """Check if approval is required"""
    if state["requires_approval"]:
        # Simulate approval process
        approved = True  # In reality, this would involve human approval
        if approved:
            result = f"APPROVED: {state['result']}"
        else:
            result = f"DENIED: {state['result']}"
    else:
        result = state["result"]
    
    return {"result": result}

def route_by_user_type(state: RoutingState) -> Literal["admin", "premium", "standard"]:
    """Route based on user type"""
    return state["user_type"]

def needs_approval(state: RoutingState) -> Literal["approval", "direct"]:
    """Check if approval is needed"""
    return "approval" if state["requires_approval"] else "direct"

# Create routing workflow
def create_routing_workflow():
    graph = StateGraph(RoutingState)
    
    # Add nodes
    graph.add_node("classify_user", classify_user)
    graph.add_node("admin_processing", admin_processing)
    graph.add_node("premium_processing", premium_processing)
    graph.add_node("standard_processing", standard_processing)
    graph.add_node("approval_check", approval_check)
    
    # Add edges
    graph.add_edge(START, "classify_user")
    
    # Conditional routing based on user type
    graph.add_conditional_edges(
        "classify_user",
        route_by_user_type,
        {
            "admin": "admin_processing",
            "premium": "premium_processing",
            "standard": "standard_processing"
        }
    )
    
    # All processing paths converge to approval check
    graph.add_conditional_edges(
        "admin_processing",
        needs_approval,
        {
            "approval": "approval_check",
            "direct": END
        }
    )
    
    graph.add_edge("premium_processing", END)
    graph.add_edge("standard_processing", END)
    graph.add_edge("approval_check", END)
    
    return graph.compile()

# Usage
routing_workflow = create_routing_workflow()
```

### 4. Loop and Iteration Patterns

Workflows that repeat until a condition is met.

```python
class IterativeState(TypedDict):
    initial_query: str
    current_query: str
    search_results: List[str]
    refined_results: List[str]
    iteration_count: int
    max_iterations: int
    quality_score: float
    target_quality: float

def search_information(state: IterativeState) -> IterativeState:
    """Search for information"""
    query = state["current_query"]
    
    # Simulate search
    mock_results = [
        f"Result 1 for: {query}",
        f"Result 2 for: {query}",
        f"Result 3 for: {query}"
    ]
    
    return {"search_results": mock_results}

def evaluate_quality(state: IterativeState) -> IterativeState:
    """Evaluate the quality of search results"""
    results = state["search_results"]
    
    # Simulate quality evaluation
    quality_score = min(0.9, 0.3 + (state["iteration_count"] * 0.2))
    
    return {"quality_score": quality_score}

def refine_query(state: IterativeState) -> IterativeState:
    """Refine the query for better results"""
    current_query = state["current_query"]
    iteration = state["iteration_count"]
    
    # Simulate query refinement
    refined_query = f"{current_query} refined_v{iteration}"
    
    return {
        "current_query": refined_query,
        "iteration_count": iteration + 1
    }

def finalize_results(state: IterativeState) -> IterativeState:
    """Finalize the search results"""
    results = state["search_results"]
    
    final_results = [f"FINAL: {result}" for result in results]
    
    return {"refined_results": final_results}

def should_continue_iteration(state: IterativeState) -> Literal["continue", "finalize"]:
    """Decide whether to continue iterating"""
    quality_met = state["quality_score"] >= state["target_quality"]
    max_iterations_reached = state["iteration_count"] >= state["max_iterations"]
    
    if quality_met or max_iterations_reached:
        return "finalize"
    else:
        return "continue"

# Create iterative workflow
def create_iterative_workflow():
    graph = StateGraph(IterativeState)
    
    # Add nodes
    graph.add_node("search_information", search_information)
    graph.add_node("evaluate_quality", evaluate_quality)
    graph.add_node("refine_query", refine_query)
    graph.add_node("finalize_results", finalize_results)
    
    # Add edges
    graph.add_edge(START, "search_information")
    graph.add_edge("search_information", "evaluate_quality")
    
    # Conditional loop
    graph.add_conditional_edges(
        "evaluate_quality",
        should_continue_iteration,
        {
            "continue": "refine_query",
            "finalize": "finalize_results"
        }
    )
    
    # Loop back to search
    graph.add_edge("refine_query", "search_information")
    graph.add_edge("finalize_results", END)
    
    return graph.compile()

# Usage
iterative_workflow = create_iterative_workflow()
result = iterative_workflow.invoke({
    "initial_query": "artificial intelligence",
    "current_query": "artificial intelligence",
    "iteration_count": 0,
    "max_iterations": 5,
    "target_quality": 0.8
})
```

## Complex Workflow Examples

### Blog Generation Workflow

From the `bloggeneration` project:

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class BlogGenerationState(TypedDict):
    topic: str
    research_data: str
    outline: str
    draft_content: str
    final_blog: str
    metadata: Dict[str, Any]

def research_topic(state: BlogGenerationState) -> BlogGenerationState:
    """Research the given topic"""
    topic = state["topic"]
    
    # Simulate research (in production, use actual search tools)
    research_data = f"""
    Research findings for: {topic}
    
    Key points:
    1. Current trends and developments
    2. Best practices and methodologies
    3. Real-world applications
    4. Future outlook
    
    Sources: Various industry publications and research papers
    """
    
    return {"research_data": research_data}

def create_outline(state: BlogGenerationState) -> BlogGenerationState:
    """Create blog outline"""
    topic = state["topic"]
    research = state["research_data"]
    
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=os.getenv("GROQ_API_KEY"))
    
    outline_prompt = ChatPromptTemplate.from_template("""
    Create a detailed outline for a blog post about: {topic}
    
    Based on this research: {research}
    
    Include:
    - Introduction
    - Main sections (3-4 sections)
    - Conclusion
    - Key takeaways
    
    Outline:
    """)
    
    outline_chain = outline_prompt | llm | StrOutputParser()
    outline = outline_chain.invoke({
        "topic": topic,
        "research": research
    })
    
    return {"outline": outline}

def write_draft(state: BlogGenerationState) -> BlogGenerationState:
    """Write the blog draft"""
    topic = state["topic"]
    outline = state["outline"]
    research = state["research_data"]
    
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=os.getenv("GROQ_API_KEY"))
    
    draft_prompt = ChatPromptTemplate.from_template("""
    Write a comprehensive blog post about: {topic}
    
    Follow this outline: {outline}
    
    Use this research: {research}
    
    Requirements:
    - Engaging introduction
    - Well-structured content
    - Practical examples
    - Professional tone
    - Clear conclusion
    
    Blog post:
    """)
    
    draft_chain = draft_prompt | llm | StrOutputParser()
    draft = draft_chain.invoke({
        "topic": topic,
        "outline": outline,
        "research": research
    })
    
    return {"draft_content": draft}

def review_and_finalize(state: BlogGenerationState) -> BlogGenerationState:
    """Review and finalize the blog"""
    draft = state["draft_content"]
    
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=os.getenv("GROQ_API_KEY"))
    
    review_prompt = ChatPromptTemplate.from_template("""
    Review and improve this blog post:
    
    {draft}
    
    Improvements to make:
    - Fix any grammatical errors
    - Improve clarity and flow
    - Add engaging elements
    - Ensure professional quality
    
    Final blog post:
    """)
    
    review_chain = review_prompt | llm | StrOutputParser()
    final_blog = review_chain.invoke({"draft": draft})
    
    # Add metadata
    metadata = {
        "word_count": len(final_blog.split()),
        "generated_at": datetime.now().isoformat(),
        "quality_score": 0.85  # Simulated quality score
    }
    
    return {
        "final_blog": final_blog,
        "metadata": metadata
    }

def create_blog_generation_workflow():
    """Create blog generation workflow"""
    graph = StateGraph(BlogGenerationState)
    
    # Add nodes
    graph.add_node("research_topic", research_topic)
    graph.add_node("create_outline", create_outline)
    graph.add_node("write_draft", write_draft)
    graph.add_node("review_and_finalize", review_and_finalize)
    
    # Add edges
    graph.add_edge(START, "research_topic")
    graph.add_edge("research_topic", "create_outline")
    graph.add_edge("create_outline", "write_draft")
    graph.add_edge("write_draft", "review_and_finalize")
    graph.add_edge("review_and_finalize", END)
    
    return graph.compile()

# Usage
blog_workflow = create_blog_generation_workflow()
result = blog_workflow.invoke({
    "topic": "The Future of Artificial Intelligence in Healthcare"
})
```

### AI News Aggregation Workflow

From the `AINEWSAgentic` project:

```python
from typing import List

class NewsAggregationState(TypedDict):
    sources: List[str]
    raw_articles: List[Dict[str, str]]
    filtered_articles: List[Dict[str, str]]
    categorized_articles: Dict[str, List[Dict[str, str]]]
    summarized_articles: List[Dict[str, str]]
    final_newsletter: str

def fetch_news(state: NewsAggregationState) -> NewsAggregationState:
    """Fetch news from multiple sources"""
    sources = state["sources"]
    
    # Simulate news fetching
    raw_articles = []
    for source in sources:
        articles = [
            {
                "title": f"AI Development News from {source}",
                "content": f"Latest AI developments from {source}...",
                "source": source,
                "timestamp": datetime.now().isoformat()
            }
            for i in range(3)  # 3 articles per source
        ]
        raw_articles.extend(articles)
    
    return {"raw_articles": raw_articles}

def filter_articles(state: NewsAggregationState) -> NewsAggregationState:
    """Filter articles based on relevance"""
    articles = state["raw_articles"]
    
    # Simulate filtering logic
    filtered_articles = []
    for article in articles:
        # Simple relevance check
        if "AI" in article["title"] or "artificial intelligence" in article["content"].lower():
            filtered_articles.append(article)
    
    return {"filtered_articles": filtered_articles}

def categorize_articles(state: NewsAggregationState) -> NewsAggregationState:
    """Categorize articles by topic"""
    articles = state["filtered_articles"]
    
    categories = {
        "technology": [],
        "business": [],
        "research": [],
        "general": []
    }
    
    for article in articles:
        # Simple categorization
        title_lower = article["title"].lower()
        if "tech" in title_lower or "development" in title_lower:
            categories["technology"].append(article)
        elif "business" in title_lower or "market" in title_lower:
            categories["business"].append(article)
        elif "research" in title_lower or "study" in title_lower:
            categories["research"].append(article)
        else:
            categories["general"].append(article)
    
    return {"categorized_articles": categories}

def summarize_articles(state: NewsAggregationState) -> NewsAggregationState:
    """Summarize articles"""
    categorized = state["categorized_articles"]
    
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=os.getenv("GROQ_API_KEY"))
    
    summarized_articles = []
    
    for category, articles in categorized.items():
        for article in articles:
            summary_prompt = ChatPromptTemplate.from_template("""
            Summarize this article in 2-3 sentences:
            
            Title: {title}
            Content: {content}
            
            Summary:
            """)
            
            summary_chain = summary_prompt | llm | StrOutputParser()
            summary = summary_chain.invoke({
                "title": article["title"],
                "content": article["content"]
            })
            
            article_summary = {
                **article,
                "summary": summary,
                "category": category
            }
            summarized_articles.append(article_summary)
    
    return {"summarized_articles": summarized_articles}

def create_newsletter(state: NewsAggregationState) -> NewsAggregationState:
    """Create final newsletter"""
    articles = state["summarized_articles"]
    
    # Group by category
    categories = {}
    for article in articles:
        category = article["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(article)
    
    # Generate newsletter
    newsletter_sections = []
    newsletter_sections.append("# AI News Weekly Summary")
    newsletter_sections.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}")
    newsletter_sections.append("")
    
    for category, category_articles in categories.items():
        newsletter_sections.append(f"## {category.title()}")
        
        for article in category_articles:
            newsletter_sections.append(f"### {article['title']}")
            newsletter_sections.append(f"*Source: {article['source']}*")
            newsletter_sections.append(f"{article['summary']}")
            newsletter_sections.append("")
    
    newsletter = "\n".join(newsletter_sections)
    
    return {"final_newsletter": newsletter}

def create_news_aggregation_workflow():
    """Create news aggregation workflow"""
    graph = StateGraph(NewsAggregationState)
    
    # Add nodes
    graph.add_node("fetch_news", fetch_news)
    graph.add_node("filter_articles", filter_articles)
    graph.add_node("categorize_articles", categorize_articles)
    graph.add_node("summarize_articles", summarize_articles)
    graph.add_node("create_newsletter", create_newsletter)
    
    # Add edges
    graph.add_edge(START, "fetch_news")
    graph.add_edge("fetch_news", "filter_articles")
    graph.add_edge("filter_articles", "categorize_articles")
    graph.add_edge("categorize_articles", "summarize_articles")
    graph.add_edge("summarize_articles", "create_newsletter")
    graph.add_edge("create_newsletter", END)
    
    return graph.compile()

# Usage
news_workflow = create_news_aggregation_workflow()
result = news_workflow.invoke({
    "sources": ["TechCrunch", "AI News", "Forbes AI"]
})
```

## Workflow Monitoring and Debugging

### Performance Monitoring

```python
import time
import psutil
from typing import Dict, Any

class WorkflowMonitor:
    def __init__(self):
        self.execution_metrics = {}
        self.node_performance = {}
        self.error_logs = []
    
    def start_execution(self, workflow_id: str):
        """Start monitoring workflow execution"""
        self.execution_metrics[workflow_id] = {
            "start_time": time.time(),
            "memory_start": psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            "nodes_executed": [],
            "errors": []
        }
    
    def log_node_execution(self, workflow_id: str, node_name: str, 
                          execution_time: float, input_size: int, output_size: int):
        """Log node execution metrics"""
        if workflow_id not in self.execution_metrics:
            return
        
        node_metric = {
            "node": node_name,
            "execution_time": execution_time,
            "input_size": input_size,
            "output_size": output_size,
            "timestamp": time.time()
        }
        
        self.execution_metrics[workflow_id]["nodes_executed"].append(node_metric)
        
        # Update node performance statistics
        if node_name not in self.node_performance:
            self.node_performance[node_name] = {
                "total_executions": 0,
                "total_time": 0,
                "avg_time": 0,
                "min_time": float('inf'),
                "max_time": 0
            }
        
        perf = self.node_performance[node_name]
        perf["total_executions"] += 1
        perf["total_time"] += execution_time
        perf["avg_time"] = perf["total_time"] / perf["total_executions"]
        perf["min_time"] = min(perf["min_time"], execution_time)
        perf["max_time"] = max(perf["max_time"], execution_time)
    
    def end_execution(self, workflow_id: str, success: bool = True):
        """End monitoring workflow execution"""
        if workflow_id not in self.execution_metrics:
            return
        
        metrics = self.execution_metrics[workflow_id]
        metrics["end_time"] = time.time()
        metrics["total_time"] = metrics["end_time"] - metrics["start_time"]
        metrics["memory_end"] = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        metrics["memory_used"] = metrics["memory_end"] - metrics["memory_start"]
        metrics["success"] = success
    
    def get_workflow_report(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution report"""
        if workflow_id not in self.execution_metrics:
            return {}
        
        metrics = self.execution_metrics[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "total_execution_time": metrics.get("total_time", 0),
            "memory_usage": metrics.get("memory_used", 0),
            "nodes_executed": len(metrics.get("nodes_executed", [])),
            "success": metrics.get("success", False),
            "node_details": metrics.get("nodes_executed", []),
            "average_node_time": sum(node["execution_time"] for node in metrics.get("nodes_executed", [])) / len(metrics.get("nodes_executed", [])) if metrics.get("nodes_executed") else 0
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        return {
            "total_workflows": len(self.execution_metrics),
            "node_performance": self.node_performance,
            "successful_workflows": sum(1 for m in self.execution_metrics.values() if m.get("success", False)),
            "failed_workflows": sum(1 for m in self.execution_metrics.values() if not m.get("success", True))
        }

# Monitored workflow wrapper
class MonitoredWorkflow:
    def __init__(self, workflow, monitor: WorkflowMonitor, workflow_id: str):
        self.workflow = workflow
        self.monitor = monitor
        self.workflow_id = workflow_id
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with monitoring"""
        self.monitor.start_execution(self.workflow_id)
        
        try:
            result = self.workflow.invoke(input_data)
            self.monitor.end_execution(self.workflow_id, success=True)
            return result
        except Exception as e:
            self.monitor.end_execution(self.workflow_id, success=False)
            raise e
    
    def get_report(self) -> Dict[str, Any]:
        """Get execution report"""
        return self.monitor.get_workflow_report(self.workflow_id)
```

### Error Handling and Recovery

```python
class ErrorState(TypedDict):
    input_data: str
    current_step: str
    error_count: int
    max_retries: int
    last_error: str
    recovery_strategy: str
    final_result: str

def resilient_processing(state: ErrorState) -> ErrorState:
    """Processing with error handling"""
    try:
        # Simulate processing that might fail
        if random.random() < 0.4:  # 40% chance of failure
            raise Exception("Processing temporarily unavailable")
        
        result = f"Successfully processed: {state['input_data']}"
        return {
            "final_result": result,
            "current_step": "completed"
        }
    except Exception as e:
        return {
            "last_error": str(e),
            "error_count": state.get("error_count", 0) + 1,
            "current_step": "error"
        }

def error_recovery(state: ErrorState) -> ErrorState:
    """Handle errors and implement recovery"""
    error_count = state["error_count"]
    max_retries = state["max_retries"]
    
    if error_count <= max_retries:
        # Implement recovery strategy
        recovery_strategy = f"retry_attempt_{error_count}"
        
        # Wait before retry (exponential backoff)
        time.sleep(min(2 ** error_count, 10))
        
        return {
            "recovery_strategy": recovery_strategy,
            "current_step": "retrying"
        }
    else:
        # Max retries reached, implement fallback
        fallback_result = f"Fallback result for: {state['input_data']}"
        return {
            "final_result": fallback_result,
            "current_step": "fallback_completed"
        }

def should_retry(state: ErrorState) -> Literal["retry", "fallback", "completed"]:
    """Determine next action based on error state"""
    if state["current_step"] == "completed":
        return "completed"
    elif state["current_step"] == "fallback_completed":
        return "completed"
    elif state["error_count"] <= state["max_retries"]:
        return "retry"
    else:
        return "fallback"

def create_resilient_workflow():
    """Create workflow with error handling"""
    graph = StateGraph(ErrorState)
    
    graph.add_node("resilient_processing", resilient_processing)
    graph.add_node("error_recovery", error_recovery)
    
    graph.add_edge(START, "resilient_processing")
    
    graph.add_conditional_edges(
        "resilient_processing",
        should_retry,
        {
            "retry": "error_recovery",
            "fallback": "error_recovery",
            "completed": END
        }
    )
    
    graph.add_edge("error_recovery", "resilient_processing")
    
    return graph.compile()
```

## Production Deployment Patterns

### Scalable Workflow Architecture

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
import json

class WorkflowOrchestrator:
    def __init__(self, max_workers=4):
        self.workflows = {}
        self.execution_queue = Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.monitor = WorkflowMonitor()
    
    def register_workflow(self, workflow_id: str, workflow_graph):
        """Register a workflow for execution"""
        self.workflows[workflow_id] = workflow_graph
    
    def submit_job(self, workflow_id: str, input_data: Dict[str, Any], 
                   execution_mode: str = "thread") -> str:
        """Submit workflow job for execution"""
        job_id = f"{workflow_id}_{int(time.time())}"
        
        if execution_mode == "thread":
            future = self.thread_pool.submit(
                self._execute_workflow, 
                workflow_id, 
                input_data, 
                job_id
            )
        else:
            future = self.process_pool.submit(
                self._execute_workflow, 
                workflow_id, 
                input_data, 
                job_id
            )
        
        return job_id
    
    def _execute_workflow(self, workflow_id: str, input_data: Dict[str, Any], 
                         job_id: str) -> Dict[str, Any]:
        """Execute workflow with monitoring"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        monitored_workflow = MonitoredWorkflow(workflow, self.monitor, job_id)
        
        return monitored_workflow.invoke(input_data)
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job execution status"""
        return self.monitor.get_workflow_report(job_id)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics"""
        return self.monitor.get_performance_summary()

# Usage
orchestrator = WorkflowOrchestrator(max_workers=8)
orchestrator.register_workflow("blog_generation", create_blog_generation_workflow())
orchestrator.register_workflow("news_aggregation", create_news_aggregation_workflow())

# Submit jobs
job1 = orchestrator.submit_job("blog_generation", {"topic": "AI in Healthcare"})
job2 = orchestrator.submit_job("news_aggregation", {"sources": ["TechCrunch", "AI News"]})

# Monitor execution
status1 = orchestrator.get_job_status(job1)
status2 = orchestrator.get_job_status(job2)
```

## Next Steps

Continue with:

- [10 - Production Deployment](10-production-deployment.md)
- [00 - Table of Contents](00-table-of-contents.md)

## Key Takeaways

- **Sequential workflows** are ideal for dependent processing steps
- **Parallel workflows** improve performance for independent tasks
- **Conditional routing** enables dynamic workflow paths
- **Iterative patterns** handle processes requiring refinement
- **Error handling** ensures robust workflow execution
- **Monitoring** provides insights into workflow performance
- **Orchestration** enables scalable workflow management
- **Recovery strategies** handle failures gracefully

---

**Remember**: Choose the right workflow pattern based on your specific requirements. Complex workflows can combine multiple patterns for optimal performance and reliability.
