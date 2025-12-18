"""Main FastAPI application for Route A."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.app.config import get_settings
from backend.app.routes import route_a
from backend.app.middleware.logging import LoggingMiddleware
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from cerebras.cloud.sdk import Cerebras

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

llm_router = APIRouter()

class LLMParseRequest(BaseModel):
    query: str
    include_reasoning: bool = False

class LLMParseResponse(BaseModel):
    indication: str
    geography: str
    confidence: str
    reasoning: str | None = None

@llm_router.post("/route-llm/parse", response_model=LLMParseResponse)
async def parse_llm_query(request: LLMParseRequest):
    """Use Cerebras to extract indication and geography from natural language"""
    
    client = Cerebras(api_key=os.environ.get("CEREB", ""))
    
    prompt = f"""You are a medical AI assistant specialized in drug repurposing. 
Extract the following information from the user's query:
1. **Indication** (disease/condition name, be specific)
2. **Geography** (country/region or "global" if not specified)

User Query: "{request.query}"

Respond ONLY in JSON format:
{{
  "indication": "<disease name>",
  "geography": "<region or global>",
  "confidence": "<high/medium/low>",
  "reasoning": "<1-2 sentence explanation>"
}}"""

    try:
        response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        
        result = response.choices[0].message.content
        
        # Parse JSON from LLM response
        import json
        import re
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in LLM response")
            
        parsed = json.loads(json_match.group())
        
        return LLMParseResponse(
            indication=parsed.get("indication", "").strip(),
            geography=parsed.get("geography", "global").strip(),
            confidence=parsed.get("confidence", "medium"),
            reasoning=parsed.get("reasoning") if request.include_reasoning else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM parsing failed: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    settings = get_settings()
    logger.info(f"Starting Route A system. Neo4j: {settings.neo4j_uri}")
    yield
    logger.info("Shutting down Route A system")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        lifespan=lifespan
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Custom logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # Routes
    app.include_router(route_a.router, prefix="/api/v1", tags=["Route A"])
    app.include_router(llm_router, prefix="/api/v1", tags=["LLM Route"])  # ✅ ADD THIS

    
    # Health check
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "route-a-system",
            "version": settings.api_version
        }
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
