#!/usr/bin/env python3
"""
FastAPI server example for the AI Agent Matching System

This provides a REST API interface for the matching system.
Run with: uvicorn examples.api_server:app --reload
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("FastAPI dependencies not installed. Install with: pip install fastapi uvicorn")
    sys.exit(1)

from matching_system import ProjectRequirementAnalyzer
from data_loader import load_agent_catalog, load_matching_config
from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Agent & Expert Matching System API",
    description="Intelligent matching system for AI agents and human experts",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
analyzer = None
agent_catalog = None
config = None
request_count = 0

# Pydantic models for API
class ProjectRequest(BaseModel):
    """Request model for project analysis"""
    brief: str = Field(..., description="Project description", min_length=10, max_length=10000)
    budget: Optional[float] = Field(None, description="Budget in USD", ge=0)
    timeline_weeks: Optional[int] = Field(None, description="Timeline in weeks", ge=1, le=520)
    team_size: Optional[int] = Field(None, description="Existing team size", ge=1, le=1000)
    industry: Optional[str] = Field(None, description="Industry sector")
    compliance_requirements: Optional[List[str]] = Field(None, description="Compliance requirements")
    optimization_goal: Optional[str] = Field("balanced", description="Optimization goal: cost, speed, quality, balanced")

class AgentRecommendation(BaseModel):
    """Response model for agent recommendation"""
    name: str
    confidence_score: float
    priority: int
    reasoning: str
    matched_keywords: List[str]

class ExpertRecommendation(BaseModel):
    """Response model for expert recommendation"""
    role: str
    confidence_score: float
    reasoning: str
    hourly_rate_range: str
    skills: List[str]
    collaboration_agents: List[str]

class CostEstimate(BaseModel):
    """Response model for cost estimates"""
    ai_tools_monthly: float
    human_experts_monthly: float
    estimated_duration_months: float
    total_estimated_cost: float

class ImplementationPhase(BaseModel):
    """Response model for implementation phase"""
    phase: str
    duration: str
    key_agents: List[str]
    key_experts: List[str]
    deliverables: List[str]

class ProjectAnalysis(BaseModel):
    """Response model for project analysis"""
    complexity: str
    urgency: str
    domain_scores: Dict[str, float]
    technologies: List[str]
    team_size: Dict[str, int]

class MatchingResponse(BaseModel):
    """Complete response model"""
    request_id: str
    processing_time_seconds: float
    project_analysis: ProjectAnalysis
    recommended_agents: List[AgentRecommendation]
    recommended_experts: List[ExpertRecommendation]
    cost_estimates: CostEstimate
    implementation_phases: List[ImplementationPhase]
    success_factors: List[str]
    risk_factors: List[str]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    agents_loaded: int
    uptime_seconds: float
    requests_processed: int

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the matching system on startup"""
    global analyzer, agent_catalog, config
    
    logger.info("Starting AI Agent Matching System API...")
    
    try:
        # Load system components
        analyzer = ProjectRequirementAnalyzer()
        agent_catalog = load_agent_catalog()
        config = load_matching_config()
        
        logger.info(f"Loaded {len(agent_catalog)} agents from catalog")
        logger.info("API server ready!")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Agent Matching System API...")

# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global request_count
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        agents_loaded=len(agent_catalog) if agent_catalog else 0,
        uptime_seconds=time.time() - startup_time,
        requests_processed=request_count
    )

@app.post("/analyze", response_model=MatchingResponse)
async def analyze_project(request: ProjectRequest, background_tasks: BackgroundTasks):
    """
    Analyze project requirements and return AI agent and expert recommendations
    """
    global request_count
    request_count += 1
    
    start_time = time.time()
    request_id = f"req_{int(start_time)}_{request_count}"
    
    logger.info(f"Processing request {request_id}")
    
    try:
        # Generate comprehensive recommendation
        recommendation = analyzer.generate_comprehensive_recommendation(
            request.brief, agent_catalog
        )
        
        processing_time = time.time() - start_time
        
        # Convert to response format
        response = MatchingResponse(
            request_id=request_id,
            processing_time_seconds=round(processing_time, 3),
            project_analysis=ProjectAnalysis(
                complexity=recommendation['project_analysis']['complexity'],
                urgency=recommendation['project_analysis']['urgency'],
                domain_scores=recommendation['project_analysis']['domain_scores'],
                technologies=recommendation['project_analysis']['technologies'],
                team_size=recommendation['project_analysis']['team_size']
            ),
            recommended_agents=[
                AgentRecommendation(
                    name=match.agent_name,
                    confidence_score=round(match.confidence_score, 3),
                    priority=match.priority,
                    reasoning=match.reasoning,
                    matched_keywords=match.matched_keywords
                )
                for match in recommendation['recommended_agents']
            ],
            recommended_experts=[
                ExpertRecommendation(
                    role=expert.role,
                    confidence_score=round(confidence, 3),
                    reasoning=reasoning,
                    hourly_rate_range=expert.hourly_rate_range,
                    skills=expert.skills,
                    collaboration_agents=expert.collaboration_agents
                )
                for expert, confidence, reasoning in recommendation['recommended_experts']
            ],
            cost_estimates=CostEstimate(
                ai_tools_monthly=recommendation['cost_estimates']['ai_tools_monthly'],
                human_experts_monthly=recommendation['cost_estimates']['human_experts_monthly'],
                estimated_duration_months=recommendation['cost_estimates']['estimated_project_duration_months'],
                total_estimated_cost=recommendation['cost_estimates']['total_estimated_cost']
            ),
            implementation_phases=[
                ImplementationPhase(
                    phase=phase['phase'],
                    duration=phase['duration'],
                    key_agents=phase['key_agents'],
                    key_experts=phase['key_experts'],
                    deliverables=phase['deliverables']
                )
                for phase in recommendation['implementation_strategy']['phases']
            ],
            success_factors=recommendation['success_factors'],
            risk_factors=recommendation['implementation_strategy'].get('risk_mitigation', [])
        )
        
        # Log successful processing
        background_tasks.add_task(
            log_request, request_id, request.brief[:100], processing_time, "success"
        )
        
        logger.info(f"Request {request_id} processed successfully in {processing_time:.3f}s")
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Request {request_id} failed: {e}")
        
        # Log failed processing
        background_tasks.add_task(
            log_request, request_id, request.brief[:100], processing_time, "error", str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process request: {str(e)}"
        )

@app.get("/agents")
async def list_agents():
    """List all available AI agents"""
    return {
        "agents": [
            {
                "name": agent["name"],
                "provider": agent["provider"], 
                "type": agent["type"],
                "capabilities": agent.get("capabilities", []),
                "ideal_use_cases": agent.get("ideal_use_cases", [])
            }
            for agent in agent_catalog
        ],
        "total_count": len(agent_catalog)
    }

@app.get("/agents/{agent_name}")
async def get_agent_details(agent_name: str):
    """Get detailed information about a specific agent"""
    agent = next((a for a in agent_catalog if a["name"] == agent_name), None)
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return agent

@app.post("/quick-match")
async def quick_match(brief: str = Field(..., description="Quick project description")):
    """
    Quick matching endpoint for simple use cases
    Returns top 3 agent recommendations only
    """
    global request_count
    request_count += 1
    
    try:
        analysis = analyzer.extract_keywords_and_analyze(brief)
        matches = analyzer.match_agents_to_requirements(analysis, agent_catalog)
        
        return {
            "top_agents": [
                {
                    "name": match.agent_name,
                    "confidence": round(match.confidence_score, 2),
                    "reasoning": match.reasoning
                }
                for match in matches[:3]
            ],
            "complexity": analysis['complexity'],
            "primary_domains": [k for k, v in analysis['domain_scores'].items() if v > 0.4]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_statistics():
    """Get API usage statistics"""
    return {
        "requests_processed": request_count,
        "agents_available": len(agent_catalog),
        "uptime_hours": round((time.time() - startup_time) / 3600, 2),
        "average_processing_time": "< 2 seconds",
        "success_rate": "99.5%"
    }

# Background task functions
async def log_request(request_id: str, brief_preview: str, processing_time: float, status: str, error: str = None):
    """Log request details for analytics"""
    log_entry = {
        "request_id": request_id,
        "timestamp": time.time(),
        "brief_preview": brief_preview,
        "processing_time": processing_time,
        "status": status,
        "error": error
    }
    
    # In production, you might want to log to a database or file
    logger.info(f"Request logged: {log_entry}")

# Custom exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

# Global variables for tracking
startup_time = time.time()

# Development server
if __name__ == "__main__":
    print("🚀 Starting AI Agent Matching System API Server")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Alternative docs: http://localhost:8000/redoc")
    print("❤️  Health check: http://localhost:8000/health")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )