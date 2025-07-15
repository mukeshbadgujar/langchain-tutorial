# 10 - Production Deployment

## Overview

This document covers production deployment strategies, best practices, and scalable architectures for LangChain and LangGraph applications based on the project implementations.

## Deployment Architecture

### 1. Microservices Architecture

Decompose monolithic applications into manageable services.

```python
# Service Base Class
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
from datetime import datetime

class BaseService(ABC):
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        self.health_status = "healthy"
        self.last_health_check = datetime.now()
    
    @abstractmethod
    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process service request"""
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        self.last_health_check = datetime.now()
        return {
            "service": self.service_name,
            "status": self.health_status,
            "timestamp": self.last_health_check.isoformat(),
            "uptime": (datetime.now() - self.last_health_check).total_seconds()
        }
    
    def log_request(self, request: Dict[str, Any], response: Dict[str, Any], 
                   duration: float):
        """Log request details"""
        self.logger.info(f"Request processed in {duration:.2f}s", extra={
            "service": self.service_name,
            "request_size": len(str(request)),
            "response_size": len(str(response)),
            "duration": duration
        })

# LLM Service
class LLMService(BaseService):
    def __init__(self, model_name: str, api_key: str):
        super().__init__("llm_service")
        self.model_name = model_name
        self.api_key = api_key
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM based on model name"""
        try:
            if "groq" in self.model_name.lower():
                from langchain_groq import ChatGroq
                return ChatGroq(model=self.model_name, groq_api_key=self.api_key)
            elif "openai" in self.model_name.lower():
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model=self.model_name, openai_api_key=self.api_key)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
        except Exception as e:
            self.health_status = "unhealthy"
            self.logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process LLM request"""
        try:
            prompt = request.get("prompt", "")
            if not prompt:
                raise ValueError("Prompt is required")
            
            start_time = datetime.now()
            response = self.llm.invoke(prompt)
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                "response": response.content,
                "model": self.model_name,
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
            
            self.log_request(request, result, duration)
            return result
        
        except Exception as e:
            self.logger.error(f"LLM processing failed: {str(e)}")
            return {
                "error": str(e),
                "service": self.service_name,
                "timestamp": datetime.now().isoformat()
            }

# Embedding Service
class EmbeddingService(BaseService):
    def __init__(self, embedding_model: str):
        super().__init__("embedding_service")
        self.embedding_model = embedding_model
        self.embeddings = self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=self.embedding_model)
        except Exception as e:
            self.health_status = "unhealthy"
            self.logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise
    
    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process embedding request"""
        try:
            texts = request.get("texts", [])
            if not texts:
                raise ValueError("Texts are required")
            
            start_time = datetime.now()
            embeddings = self.embeddings.embed_documents(texts)
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                "embeddings": embeddings,
                "model": self.embedding_model,
                "text_count": len(texts),
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
            
            self.log_request(request, result, duration)
            return result
        
        except Exception as e:
            self.logger.error(f"Embedding processing failed: {str(e)}")
            return {
                "error": str(e),
                "service": self.service_name,
                "timestamp": datetime.now().isoformat()
            }

# Vector Store Service
class VectorStoreService(BaseService):
    def __init__(self, store_type: str, connection_params: Dict[str, Any]):
        super().__init__("vectorstore_service")
        self.store_type = store_type
        self.connection_params = connection_params
        self.vectorstore = self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize vector store"""
        try:
            if self.store_type == "faiss":
                from langchain_community.vectorstores import FAISS
                from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                return FAISS.from_texts(["dummy"], embeddings)
            elif self.store_type == "chroma":
                from langchain_community.vectorstores import Chroma
                return Chroma(persist_directory=self.connection_params.get("persist_directory"))
            else:
                raise ValueError(f"Unsupported vector store: {self.store_type}")
        except Exception as e:
            self.health_status = "unhealthy"
            self.logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process vector store request"""
        try:
            operation = request.get("operation", "")
            
            if operation == "add":
                return self._add_documents(request)
            elif operation == "search":
                return self._search_documents(request)
            elif operation == "delete":
                return self._delete_documents(request)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        
        except Exception as e:
            self.logger.error(f"Vector store operation failed: {str(e)}")
            return {
                "error": str(e),
                "service": self.service_name,
                "timestamp": datetime.now().isoformat()
            }
    
    def _add_documents(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Add documents to vector store"""
        documents = request.get("documents", [])
        metadatas = request.get("metadatas", [])
        
        start_time = datetime.now()
        self.vectorstore.add_texts(documents, metadatas=metadatas)
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "operation": "add",
            "documents_added": len(documents),
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
    
    def _search_documents(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Search documents in vector store"""
        query = request.get("query", "")
        k = request.get("k", 5)
        
        start_time = datetime.now()
        results = self.vectorstore.similarity_search(query, k=k)
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "operation": "search",
            "query": query,
            "results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in results],
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
    
    def _delete_documents(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Delete documents from vector store"""
        document_ids = request.get("document_ids", [])
        
        start_time = datetime.now()
        # Implementation depends on vector store capabilities
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "operation": "delete",
            "documents_deleted": len(document_ids),
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }

# Service Registry
class ServiceRegistry:
    def __init__(self):
        self.services = {}
        self.logger = logging.getLogger("service_registry")
    
    def register_service(self, service_name: str, service_instance: BaseService):
        """Register a service"""
        self.services[service_name] = service_instance
        self.logger.info(f"Service registered: {service_name}")
    
    def get_service(self, service_name: str) -> Optional[BaseService]:
        """Get a service instance"""
        return self.services.get(service_name)
    
    def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all services"""
        health_results = {}
        for service_name, service in self.services.items():
            health_results[service_name] = service.health_check()
        
        return {
            "overall_status": "healthy" if all(
                result["status"] == "healthy" 
                for result in health_results.values()
            ) else "unhealthy",
            "services": health_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def list_services(self) -> List[str]:
        """List all registered services"""
        return list(self.services.keys())

# Initialize services
def initialize_services() -> ServiceRegistry:
    """Initialize all services"""
    registry = ServiceRegistry()
    
    # Initialize LLM service
    llm_service = LLMService(
        model_name="Gemma2-9b-It",
        api_key=os.getenv("GROQ_API_KEY")
    )
    registry.register_service("llm", llm_service)
    
    # Initialize embedding service
    embedding_service = EmbeddingService(
        embedding_model="text-embedding-ada-002"
    )
    registry.register_service("embeddings", embedding_service)
    
    # Initialize vector store service
    vectorstore_service = VectorStoreService(
        store_type="faiss",
        connection_params={}
    )
    registry.register_service("vectorstore", vectorstore_service)
    
    return registry
```

### 2. FastAPI Production Server

Based on the `bloggeneration` FastAPI implementation:

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import asyncio
from contextlib import asynccontextmanager

# Request/Response Models
class BlogRequest(BaseModel):
    topic: str
    target_audience: Optional[str] = "general"
    word_count: Optional[int] = 1000
    tone: Optional[str] = "professional"
    include_examples: Optional[bool] = True

class BlogResponse(BaseModel):
    blog_id: str
    content: str
    metadata: Dict[str, Any]
    generation_time: float
    word_count: int

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, Any]
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str

# Global service registry
service_registry = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    global service_registry
    service_registry = initialize_services()
    yield
    # Shutdown
    # Cleanup resources if needed
    pass

# Create FastAPI app
app = FastAPI(
    title="LangChain Production API",
    description="Production-ready API for LangChain applications",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

def authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Authenticate requests"""
    # Implement your authentication logic here
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return credentials.credentials

# Dependency to get service registry
def get_service_registry():
    return service_registry

# Blog Generation Endpoint
@app.post("/generate-blog", response_model=BlogResponse)
async def generate_blog(
    request: BlogRequest,
    background_tasks: BackgroundTasks,
    registry: ServiceRegistry = Depends(get_service_registry),
    token: str = Depends(authenticate)
):
    """Generate blog content"""
    try:
        # Get LLM service
        llm_service = registry.get_service("llm")
        if not llm_service:
            raise HTTPException(status_code=503, detail="LLM service unavailable")
        
        # Generate blog content
        blog_prompt = f"""
        Write a {request.word_count}-word blog post about: {request.topic}
        
        Target audience: {request.target_audience}
        Tone: {request.tone}
        Include examples: {request.include_examples}
        
        Requirements:
        - Engaging introduction
        - Well-structured content
        - {request.tone} tone
        - Approximately {request.word_count} words
        """
        
        start_time = time.time()
        
        llm_response = llm_service.process({"prompt": blog_prompt})
        
        if "error" in llm_response:
            raise HTTPException(status_code=500, detail=llm_response["error"])
        
        generation_time = time.time() - start_time
        content = llm_response["response"]
        word_count = len(content.split())
        
        blog_id = f"blog_{int(time.time())}"
        
        # Log generation for analytics
        background_tasks.add_task(
            log_blog_generation,
            blog_id,
            request.topic,
            word_count,
            generation_time
        )
        
        return BlogResponse(
            blog_id=blog_id,
            content=content,
            metadata={
                "topic": request.topic,
                "target_audience": request.target_audience,
                "requested_word_count": request.word_count,
                "actual_word_count": word_count,
                "tone": request.tone,
                "model": llm_response.get("model", "unknown")
            },
            generation_time=generation_time,
            word_count=word_count
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Vector Search Endpoint
@app.post("/search")
async def search_documents(
    query: str,
    k: int = 5,
    registry: ServiceRegistry = Depends(get_service_registry),
    token: str = Depends(authenticate)
):
    """Search documents using vector similarity"""
    try:
        vectorstore_service = registry.get_service("vectorstore")
        if not vectorstore_service:
            raise HTTPException(status_code=503, detail="Vector store service unavailable")
        
        search_request = {
            "operation": "search",
            "query": query,
            "k": k
        }
        
        response = vectorstore_service.process(search_request)
        
        if "error" in response:
            raise HTTPException(status_code=500, detail=response["error"])
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Health Check Endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check(registry: ServiceRegistry = Depends(get_service_registry)):
    """Check system health"""
    try:
        health_data = registry.health_check_all()
        return HealthResponse(**health_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Metrics Endpoint
@app.get("/metrics")
async def get_metrics(registry: ServiceRegistry = Depends(get_service_registry)):
    """Get system metrics"""
    try:
        metrics = {
            "services": registry.list_services(),
            "health": registry.health_check_all(),
            "system": {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        }
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

# Background task for logging
async def log_blog_generation(blog_id: str, topic: str, word_count: int, generation_time: float):
    """Log blog generation for analytics"""
    # Implementation for logging/analytics
    logging.info(f"Blog generated: {blog_id}, Topic: {topic}, Words: {word_count}, Time: {generation_time:.2f}s")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="Not Found",
            message="The requested resource was not found",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

# Run server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable in production
        workers=4,
        log_level="info"
    )
```

### 3. Streamlit Production App

Based on the `AINEWSAgentic` Streamlit implementation:

```python
import streamlit as st
import asyncio
from typing import Dict, Any
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="LangChain Production Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-healthy {
        color: #4CAF50;
    }
    .status-unhealthy {
        color: #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'service_registry' not in st.session_state:
    st.session_state.service_registry = initialize_services()

# Sidebar
with st.sidebar:
    st.header("🛠️ System Controls")
    
    # Service status
    st.subheader("Service Status")
    health_data = st.session_state.service_registry.health_check_all()
    
    for service_name, health in health_data["services"].items():
        status_class = "status-healthy" if health["status"] == "healthy" else "status-unhealthy"
        st.markdown(f"**{service_name}**: <span class='{status_class}'>{health['status']}</span>", 
                   unsafe_allow_html=True)
    
    # Refresh button
    if st.button("🔄 Refresh Status"):
        st.rerun()

# Main content
st.markdown("<h1 class='main-header'>🚀 LangChain Production Dashboard</h1>", 
           unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📝 Blog Generation", "🔍 Search", "📈 Analytics"])

with tab1:
    st.header("System Overview")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Services", len(health_data["services"]))
    
    with col2:
        healthy_services = sum(1 for s in health_data["services"].values() if s["status"] == "healthy")
        st.metric("Healthy Services", healthy_services)
    
    with col3:
        # Simulate some metrics
        st.metric("Requests Today", "1,234", "↗️ 12%")
    
    with col4:
        st.metric("Avg Response Time", "245ms", "↘️ 5%")
    
    # Service health visualization
    st.subheader("Service Health Status")
    
    # Create health data for visualization
    health_df = pd.DataFrame([
        {"Service": name, "Status": health["status"], "Uptime": health.get("uptime", 0)}
        for name, health in health_data["services"].items()
    ])
    
    # Health status chart
    fig_health = px.bar(
        health_df,
        x="Service",
        y="Uptime",
        color="Status",
        title="Service Health and Uptime"
    )
    st.plotly_chart(fig_health, use_container_width=True)

with tab2:
    st.header("📝 Blog Generation")
    
    # Blog generation form
    with st.form("blog_generation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input("Blog Topic", placeholder="Enter your blog topic...")
            target_audience = st.selectbox("Target Audience", 
                                         ["general", "technical", "business", "academic"])
            word_count = st.slider("Word Count", min_value=300, max_value=2000, value=1000)
        
        with col2:
            tone = st.selectbox("Tone", ["professional", "casual", "formal", "conversational"])
            include_examples = st.checkbox("Include Examples", value=True)
        
        submit_button = st.form_submit_button("Generate Blog")
    
    if submit_button and topic:
        with st.spinner("Generating blog content..."):
            try:
                # Get LLM service
                llm_service = st.session_state.service_registry.get_service("llm")
                
                if llm_service:
                    blog_prompt = f"""
                    Write a {word_count}-word blog post about: {topic}
                    
                    Target audience: {target_audience}
                    Tone: {tone}
                    Include examples: {include_examples}
                    
                    Requirements:
                    - Engaging introduction
                    - Well-structured content
                    - {tone} tone
                    - Approximately {word_count} words
                    """
                    
                    start_time = time.time()
                    response = llm_service.process({"prompt": blog_prompt})
                    generation_time = time.time() - start_time
                    
                    if "error" not in response:
                        st.success(f"Blog generated successfully in {generation_time:.2f} seconds!")
                        
                        # Display generated content
                        st.subheader("Generated Blog Content")
                        st.markdown(response["response"])
                        
                        # Show metadata
                        with st.expander("Generation Details"):
                            st.json({
                                "topic": topic,
                                "word_count": len(response["response"].split()),
                                "generation_time": f"{generation_time:.2f}s",
                                "model": response.get("model", "unknown")
                            })
                    else:
                        st.error(f"Error generating blog: {response['error']}")
                else:
                    st.error("LLM service is not available")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

with tab3:
    st.header("🔍 Document Search")
    
    # Search form
    with st.form("search_form"):
        query = st.text_input("Search Query", placeholder="Enter your search query...")
        k = st.slider("Number of Results", min_value=1, max_value=20, value=5)
        
        search_button = st.form_submit_button("Search")
    
    if search_button and query:
        with st.spinner("Searching documents..."):
            try:
                vectorstore_service = st.session_state.service_registry.get_service("vectorstore")
                
                if vectorstore_service:
                    search_request = {
                        "operation": "search",
                        "query": query,
                        "k": k
                    }
                    
                    response = vectorstore_service.process(search_request)
                    
                    if "error" not in response:
                        st.success(f"Found {len(response['results'])} results")
                        
                        # Display results
                        for i, result in enumerate(response["results"], 1):
                            with st.expander(f"Result {i}"):
                                st.markdown(result["content"])
                                if result["metadata"]:
                                    st.json(result["metadata"])
                    else:
                        st.error(f"Search failed: {response['error']}")
                else:
                    st.error("Vector store service is not available")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

with tab4:
    st.header("📈 Analytics")
    
    # Generate sample analytics data
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    analytics_data = pd.DataFrame({
        'Date': dates,
        'Requests': np.random.randint(50, 200, len(dates)),
        'Response_Time': np.random.uniform(100, 400, len(dates)),
        'Success_Rate': np.random.uniform(0.95, 1.0, len(dates))
    })
    
    # Request volume chart
    fig_requests = px.line(
        analytics_data,
        x='Date',
        y='Requests',
        title='Daily Request Volume'
    )
    st.plotly_chart(fig_requests, use_container_width=True)
    
    # Response time and success rate
    col1, col2 = st.columns(2)
    
    with col1:
        fig_response = px.line(
            analytics_data,
            x='Date',
            y='Response_Time',
            title='Average Response Time (ms)'
        )
        st.plotly_chart(fig_response, use_container_width=True)
    
    with col2:
        fig_success = px.line(
            analytics_data,
            x='Date',
            y='Success_Rate',
            title='Success Rate'
        )
        st.plotly_chart(fig_success, use_container_width=True)
    
    # Performance metrics table
    st.subheader("Performance Summary")
    summary_data = {
        "Metric": ["Total Requests", "Average Response Time", "Success Rate", "Uptime"],
        "Value": ["45,678", "245ms", "99.2%", "99.9%"],
        "Change": ["+12%", "-5%", "+0.3%", "0%"]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**LangChain Production Dashboard** - Built with Streamlit and LangChain")
```

## Container Deployment

### 1. Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Main application
  langchain-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://postgres:password@postgres:5432/langchain
    depends_on:
      - redis
      - postgres
    volumes:
      - ./logs:/app/logs
    networks:
      - langchain-network
    restart: unless-stopped

  # Redis for caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - langchain-network
    restart: unless-stopped

  # PostgreSQL for data storage
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=langchain
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - langchain-network
    restart: unless-stopped

  # Vector database (Weaviate)
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
    environment:
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
    volumes:
      - weaviate_data:/var/lib/weaviate
    networks:
      - langchain-network
    restart: unless-stopped

  # Monitoring (Prometheus)
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - langchain-network
    restart: unless-stopped

  # Monitoring (Grafana)
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - langchain-network
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
  weaviate_data:
  prometheus_data:
  grafana_data:

networks:
  langchain-network:
    driver: bridge
```

### 3. Kubernetes Deployment

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langchain-app
  labels:
    app: langchain-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langchain-app
  template:
    metadata:
      labels:
        app: langchain-app
    spec:
      containers:
      - name: langchain-app
        image: langchain-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: GROQ_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: groq-api-key
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-api-key
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: langchain-app-service
spec:
  selector:
    app: langchain-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
type: Opaque
stringData:
  groq-api-key: "your-groq-api-key"
  openai-api-key: "your-openai-api-key"
```

## Performance Optimization

### 1. Caching Strategy

```python
import redis
import json
from functools import wraps
from typing import Any, Callable, Optional

class CacheManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 hour
    
    def cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key"""
        key_parts = [prefix]
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        return ":".join(key_parts)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logging.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or self.default_ttl
            serialized_value = json.dumps(value, default=str)
            return self.redis_client.setex(key, ttl, serialized_value)
        except Exception as e:
            logging.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            return self.redis_client.delete(key)
        except Exception as e:
            logging.error(f"Cache delete error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logging.error(f"Cache clear error: {e}")
            return 0

# Cache decorator
def cached(prefix: str, ttl: Optional[int] = None):
    """Decorator to cache function results"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_manager = CacheManager()
            
            # Generate cache key
            cache_key = cache_manager.cache_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# Usage example
@cached("llm_response", ttl=1800)  # Cache for 30 minutes
def generate_llm_response(prompt: str, model: str) -> str:
    """Generate LLM response with caching"""
    llm_service = service_registry.get_service("llm")
    response = llm_service.process({"prompt": prompt})
    return response["response"]
```

### 2. Connection Pooling

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600
        )
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: Optional[Dict] = None):
        """Execute database query"""
        with self.get_connection() as conn:
            return conn.execute(query, params or {})

# HTTP client with connection pooling
import aiohttp
import asyncio

class HTTPClientManager:
    def __init__(self):
        self.session = None
        self.connector = None
    
    async def __aenter__(self):
        self.connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()

# Usage
async def make_api_calls():
    async with HTTPClientManager() as session:
        # Make multiple API calls with connection pooling
        tasks = [
            session.get("https://api.example.com/endpoint1"),
            session.get("https://api.example.com/endpoint2"),
            session.get("https://api.example.com/endpoint3")
        ]
        responses = await asyncio.gather(*tasks)
        return responses
```

### 3. Load Balancing

```python
import random
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_RANDOM = "weighted_random"
    LEAST_CONNECTIONS = "least_connections"
    HEALTH_BASED = "health_based"

@dataclass
class ServiceInstance:
    id: str
    host: str
    port: int
    weight: float = 1.0
    active_connections: int = 0
    health_score: float = 1.0
    is_healthy: bool = True

class LoadBalancer:
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.instances: List[ServiceInstance] = []
        self.current_index = 0
    
    def add_instance(self, instance: ServiceInstance):
        """Add service instance"""
        self.instances.append(instance)
    
    def remove_instance(self, instance_id: str):
        """Remove service instance"""
        self.instances = [i for i in self.instances if i.id != instance_id]
    
    def get_healthy_instances(self) -> List[ServiceInstance]:
        """Get only healthy instances"""
        return [i for i in self.instances if i.is_healthy]
    
    def select_instance(self) -> Optional[ServiceInstance]:
        """Select instance based on strategy"""
        healthy_instances = self.get_healthy_instances()
        
        if not healthy_instances:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
            return self._weighted_random_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.HEALTH_BASED:
            return self._health_based_selection(healthy_instances)
        
        return healthy_instances[0]
    
    def _round_robin_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round robin selection"""
        instance = instances[self.current_index % len(instances)]
        self.current_index += 1
        return instance
    
    def _weighted_random_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted random selection"""
        total_weight = sum(i.weight for i in instances)
        random_value = random.uniform(0, total_weight)
        
        current_weight = 0
        for instance in instances:
            current_weight += instance.weight
            if random_value <= current_weight:
                return instance
        
        return instances[-1]
    
    def _least_connections_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least connections selection"""
        return min(instances, key=lambda x: x.active_connections)
    
    def _health_based_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Health-based selection"""
        return max(instances, key=lambda x: x.health_score)
    
    def update_instance_health(self, instance_id: str, health_score: float, is_healthy: bool):
        """Update instance health"""
        for instance in self.instances:
            if instance.id == instance_id:
                instance.health_score = health_score
                instance.is_healthy = is_healthy
                break
    
    def increment_connections(self, instance_id: str):
        """Increment active connections"""
        for instance in self.instances:
            if instance.id == instance_id:
                instance.active_connections += 1
                break
    
    def decrement_connections(self, instance_id: str):
        """Decrement active connections"""
        for instance in self.instances:
            if instance.id == instance_id:
                instance.active_connections = max(0, instance.active_connections - 1)
                break

# Load balanced service client
class LoadBalancedServiceClient:
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
    
    async def make_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to load balanced service"""
        instance = self.load_balancer.select_instance()
        
        if not instance:
            raise Exception("No healthy instances available")
        
        # Increment connection count
        self.load_balancer.increment_connections(instance.id)
        
        try:
            # Make actual request to instance
            response = await self._make_http_request(instance, request_data)
            return response
        finally:
            # Decrement connection count
            self.load_balancer.decrement_connections(instance.id)
    
    async def _make_http_request(self, instance: ServiceInstance, 
                                request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to specific instance"""
        # Implementation of HTTP request
        # This would typically use aiohttp or similar
        pass
```

## Monitoring and Observability

### 1. Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
REQUEST_COUNT = Counter('langchain_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('langchain_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('langchain_active_connections', 'Active connections')
MODEL_USAGE = Counter('langchain_model_usage_total', 'Model usage', ['model_name'])

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            # Get request info
            method = scope["method"]
            path = scope["path"]
            
            # Increment request counter
            REQUEST_COUNT.labels(method=method, endpoint=path).inc()
            
            # Track active connections
            ACTIVE_CONNECTIONS.inc()
            
            try:
                await self.app(scope, receive, send)
            finally:
                # Record duration
                duration = time.time() - start_time
                REQUEST_DURATION.observe(duration)
                
                # Decrement active connections
                ACTIVE_CONNECTIONS.dec()
        else:
            await self.app(scope, receive, send)

# Start metrics server
start_http_server(8001)
```

### 2. Logging Configuration

```python
import logging
import sys
from datetime import datetime
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

def setup_logging():
    """Setup centralized logging"""
    
    # Create formatter
    formatter = JSONFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Specific loggers
    langchain_logger = logging.getLogger("langchain")
    langchain_logger.setLevel(logging.DEBUG)
    
    return root_logger

# Usage
logger = setup_logging()
```

### 3. Health Checks

```python
from typing import Dict, List, Callable
from dataclasses import dataclass
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"

@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    message: str
    duration: float
    details: Dict[str, Any] = None

class HealthChecker:
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
    
    def register_check(self, name: str, check_func: Callable):
        """Register health check"""
        self.checks[name] = check_func
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run individual health check"""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Check not found",
                duration=0.0
            )
        
        start_time = time.time()
        try:
            result = await self.checks[name]()
            duration = time.time() - start_time
            
            return HealthCheckResult(
                name=name,
                status=HealthStatus.HEALTHY,
                message="OK",
                duration=duration,
                details=result
            )
        except Exception as e:
            duration = time.time() - start_time
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration=duration
            )
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks"""
        results = {}
        
        for name in self.checks:
            results[name] = await self.run_check(name)
        
        return results
    
    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Get overall system health"""
        statuses = [result.status for result in results.values()]
        
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED

# Health check implementations
async def check_database():
    """Check database connectivity"""
    try:
        # Test database connection
        db_manager = DatabaseManager("postgresql://...")
        with db_manager.get_connection() as conn:
            conn.execute("SELECT 1")
        return {"connection": "ok"}
    except Exception as e:
        raise Exception(f"Database check failed: {str(e)}")

async def check_redis():
    """Check Redis connectivity"""
    try:
        cache_manager = CacheManager()
        cache_manager.redis_client.ping()
        return {"connection": "ok"}
    except Exception as e:
        raise Exception(f"Redis check failed: {str(e)}")

async def check_external_api():
    """Check external API availability"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.openai.com/v1/models") as response:
                if response.status == 200:
                    return {"api": "ok", "status_code": response.status}
                else:
                    raise Exception(f"API returned status {response.status}")
    except Exception as e:
        raise Exception(f"External API check failed: {str(e)}")

# Setup health checker
health_checker = HealthChecker()
health_checker.register_check("database", check_database)
health_checker.register_check("redis", check_redis)
health_checker.register_check("external_api", check_external_api)
```

## Security Best Practices

### 1. API Security

```python
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
import secrets

class SecurityManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = "HS256"
        self.token_expiry = timedelta(hours=24)
    
    def generate_token(self, user_id: str, scopes: List[str]) -> str:
        """Generate JWT token"""
        payload = {
            "user_id": user_id,
            "scopes": scopes,
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow(),
            "jti": secrets.token_hex(16)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def require_scope(self, required_scope: str):
        """Decorator to require specific scope"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Get token from request
                token = kwargs.get("token")
                if not token:
                    raise HTTPException(status_code=401, detail="Token required")
                
                payload = self.verify_token(token)
                scopes = payload.get("scopes", [])
                
                if required_scope not in scopes:
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator

# Rate limiting
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        
        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < self.window_seconds
        ]
        
        # Check if limit exceeded
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True

# Usage
security_manager = SecurityManager(os.getenv("SECRET_KEY"))
rate_limiter = RateLimiter(max_requests=100, window_seconds=3600)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    response = await call_next(request)
    return response
```

### 2. Data Protection

```python
from cryptography.fernet import Fernet
import base64
import hashlib

class DataProtection:
    def __init__(self, encryption_key: str):
        # Generate key from password
        key = base64.urlsafe_b64encode(
            hashlib.sha256(encryption_key.encode()).digest()
        )
        self.cipher = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        import bcrypt
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password"""
        import bcrypt
        return bcrypt.checkpw(password.encode(), hashed.encode())

# Input validation
from pydantic import BaseModel, validator
import re

class SecureInput(BaseModel):
    content: str
    
    @validator('content')
    def validate_content(cls, v):
        # Check for SQL injection patterns
        sql_patterns = [
            r"(\bSELECT\b|\bUNION\b|\bDROP\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b)",
            r"(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+",
            r"['\";]"
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Invalid input detected")
        
        # Check for XSS patterns
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*="
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Invalid input detected")
        
        return v

# Usage
data_protection = DataProtection(os.getenv("ENCRYPTION_KEY"))
```

## Key Takeaways

- **Microservices architecture** enables scalable and maintainable applications
- **Container deployment** provides consistent environments across development and production
- **Performance optimization** through caching, connection pooling, and load balancing
- **Monitoring and observability** are essential for production systems
- **Security** must be implemented at every layer
- **Health checks** ensure system reliability
- **Load balancing** distributes traffic efficiently
- **Proper logging** aids in debugging and monitoring

---

**Remember**: Production deployment requires careful consideration of security, scalability, monitoring, and reliability. Always test thoroughly in staging environments before deploying to production.

This completes the comprehensive documentation for the LangChain and LangGraph project. The documentation covers everything from basic concepts to advanced production deployment strategies, providing a complete learning resource for understanding and implementing LangChain applications.
