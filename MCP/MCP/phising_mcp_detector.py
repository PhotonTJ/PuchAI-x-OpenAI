import asyncio
import os
import pickle
import re
import logging
import json
from datetime import datetime
from typing import Annotated, Optional, Tuple
from urllib.parse import urlparse
import validators
from dotenv import load_dotenv
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# FastMCP imports
from fastmcp import FastMCP
from pydantic import Field, BaseModel, field_validator

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phishing_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Environment Configuration ---


TOKEN = "superrandomstring"  # Change this to something random!
MY_NUMBER = "917498124973"  # Your WhatsApp number for Puch AI validation

# Optional Settings
DEFAULT_LAMBDA = 0.6  # Weight for ML vs Grok (0.6 = 60% Grok, 40% ML)
MODEL_PATH = "phishing_mcp_server.log"  # Path to your trained model
VECTORIZER_PATH = "tfidf_vectorizer.pkl"  # Path to your vectorizer

# Grok AI Settings (Optional - leave empty if you don't have Grok API)
GROK_API_URL = ""  # Example: "https://api.x.ai/v1/chat/completions"
GROK_API_KEY = ""  # Your Grok API key

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MCP_PORT = int(os.getenv("MCP_PORT", "8001"))

# Constants
PHISHING_THRESHOLD = 0.5
HIGH_RISK_THRESHOLD = 0.8
MEDIUM_RISK_THRESHOLD = 0.6
MAX_BATCH_SIZE = 10

logger.info("✅ Configuration loaded successfully")

# Global variables
model = None
vectorizer = None
server_stats = {
    "startup_time": datetime.now(),
    "total_predictions": 0,
    "phishing_detected": 0,
    "errors": 0,
    "user_requests": {}
}

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Phishing Detection API",
    description="API for detecting phishing websites",
    version="1.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize MCP Server ---
mcp = FastMCP("Phishing Detection MCP Server")

# --- Load Models ---
def load_models():
    """Load ML model and vectorizer with error handling"""
    global model, vectorizer
    
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            logger.info(f"✅ Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"⚠️ Model file not found at {MODEL_PATH}")
            
        if os.path.exists(VECTORIZER_PATH):
            with open(VECTORIZER_PATH, "rb") as f:
                vectorizer = pickle.load(f)
            logger.info(f"✅ Vectorizer loaded from {VECTORIZER_PATH}")
        else:
            logger.warning(f"⚠️ Vectorizer file not found at {VECTORIZER_PATH}")
            
        if model and vectorizer:
            logger.info("✅ All models loaded successfully")
        else:
            logger.warning("⚠️ Some models missing - using fallback scores")
            
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        model = None
        vectorizer = None

load_models()

# --- Utility Functions ---
def _now() -> str:
    return datetime.utcnow().isoformat()

def get_confidence_level(score: float) -> str:
    """Determine confidence level based on score"""
    if score <= 0.2 or score >= 0.8:
        return "high"
    elif score <= 0.35 or score >= 0.65:
        return "medium"
    else:
        return "low"

def preprocess_url(url: str) -> str:
    """Clean and preprocess URL for ML model"""
    try:
        url = url.lower().strip()
        url = re.sub(r"https?://", "", url)
        url = re.sub(r"www\.", "", url)
        url = re.sub(r"/$", "", url)
        return url
    except Exception as e:
        logger.warning(f"URL preprocessing error: {e}")
        return url

def validate_url(url: str) -> bool:
    """Validate if the provided string is a valid URL"""
    try:
        if not url or not url.strip():
            return False
        
        url = url.strip()
        if validators.url(url) or validators.domain(url):
            return True
            
        if not url.startswith(('http://', 'https://')):
            test_url = f"https://{url}"
            return validators.url(test_url)
            
        return False
    except Exception:
        return False

def ml_model_score(url: str) -> Tuple[float, bool]:
    """Get phishing probability from ML model"""
    try:
        if not model or not vectorizer:
            logger.warning("ML model not available, using fallback score")
            return 0.5, False
        
        clean_url = preprocess_url(url)
        features = vectorizer.transform([clean_url])
        probability = float(model.predict_proba(features)[0][1])
        
        return max(0.0, min(1.0, probability)), True
        
    except Exception as e:
        logger.error(f"ML model prediction error: {e}")
        return 0.5, False

async def groq_score(url: str, timeout: int = 10) -> Tuple[float, bool]:
    """Get phishing probability from Groq API"""
    try:
        if not GROQ_API_URL or not GROQ_API_KEY:
            logger.warning("Groq API not configured, using fallback score")
            return 0.5, False
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a cybersecurity expert. Analyze URLs for phishing indicators and respond with ONLY a probability score between 0.0 and 1.0, where 0.0 is completely safe and 1.0 is definitely phishing."
                },
                {
                    "role": "user", 
                    "content": f"Analyze this URL for phishing indicators: {url}"
                }
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GROQ_API_URL, 
                json=payload, 
                headers=headers, 
                timeout=timeout
            )
            response.raise_for_status()
            
            response_data = response.json()
            output_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "0.5").strip()
            
            score_match = re.search(r"0?\.\d+|[01]\.?\d*", output_text)
            if score_match:
                score = float(score_match.group())
                return max(0.0, min(1.0, score)), True
            else:
                return 0.5, False
                
    except httpx.RequestError as e:
        logger.error(f"Groq API request error: {e}")
        return 0.5, False
    except Exception as e:
        logger.error(f"Groq API unexpected error: {e}")
        return 0.5, False

def update_user_stats(user_id: str, is_phishing: bool):
    """Update statistics for user and global"""
    server_stats["total_predictions"] += 1
    if is_phishing:
        server_stats["phishing_detected"] += 1
    
    if user_id not in server_stats["user_requests"]:
        server_stats["user_requests"][user_id] = {
            "total": 0,
            "phishing_found": 0,
            "first_request": _now(),
            "last_request": _now()
        }
    
    user_stats = server_stats["user_requests"][user_id]
    user_stats["total"] += 1
    if is_phishing:
        user_stats["phishing_found"] += 1
    user_stats["last_request"] = _now()

async def analyze_url_core(url: str, user_id: str = "api_user", lambda_override: Optional[float] = None):
    """Core analysis function used by both MCP and FastAPI"""
    try:
        start_time = datetime.now()
        
        if not validate_url(url):
            return {"error": "Invalid URL format", "success": False}
        
        lambda_val = lambda_override if lambda_override is not None else DEFAULT_LAMBDA
        
        # Get predictions
        ml_score, ml_success = ml_model_score(url)
        groq_score_val, groq_success = await groq_score(url)
        
        # Calculate final score
        if ml_success and groq_success:
            final_score = lambda_val * groq_score_val + (1 - lambda_val) * ml_score
        elif ml_success:
            final_score = ml_score
        elif groq_success:
            final_score = groq_score_val
        else:
            final_score = 0.5
        
        final_score = max(0.0, min(1.0, final_score))
        is_phishing = final_score > PHISHING_THRESHOLD
        confidence = get_confidence_level(final_score)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update stats
        update_user_stats(user_id, is_phishing)
        
        # Log
        logger.info(f"User {user_id}: {url} -> {final_score:.3f} ({'PHISHING' if is_phishing else 'SAFE'})")
        
        return {
            "success": True,
            "url": url,
            "is_phishing": is_phishing,
            "risk_level": "HIGH" if final_score >= HIGH_RISK_THRESHOLD else "MEDIUM" if final_score >= MEDIUM_RISK_THRESHOLD else "LOW",
            "confidence": confidence,
            "final_score": round(final_score, 4),
            "model_scores": {
                "ml_model": round(ml_score, 4) if ml_success else None,
                "groq_ai": round(groq_score_val, 4) if groq_success else None,
            },
            "metadata": {
                "processing_time_ms": round(processing_time, 2),
                "timestamp": _now(),
                "models_used": {
                    "ml_model": ml_success,
                    "groq_ai": groq_success
                }
            }
        }
        
    except Exception as e:
        server_stats["errors"] += 1
        logger.error(f"Analysis error: {e}")
        return {"error": str(e), "success": False}

# --- FastAPI Endpoints ---

class AnalyzeRequest(BaseModel):
    url: str
    user_id: Optional[str] = "web_user"
    lambda_override: Optional[float] = None

    @field_validator("url")
    @classmethod
    def normalize_url(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("URL cannot be empty")
        if not re.match(r"^https?://", v):
            v = "https://" + v
        return v

class BatchAnalyzeRequest(BaseModel):
    urls: list[str]
    user_id: Optional[str] = "web_user"

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🔒 Phishing Detection API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /analyze - Analyze single URL",
            "batch": "POST /batch-analyze - Analyze multiple URLs",
            "stats": "GET /stats/{user_id} - Get user statistics",
            "health": "GET /health - Server health check"
        },
        "mcp_server": f"MCP server running on port {MCP_PORT}",
        "docs": "/docs"
    }

@app.post("/analyze")
async def analyze_website(request: AnalyzeRequest):
    """Analyze a single website for phishing - Main endpoint for frontend"""
    try:
        result = await analyze_url_core(
            url=request.url,
            user_id=request.user_id,
            lambda_override=request.lambda_override
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Format response like WhatsApp bot
        if result["is_phishing"]:
            message = f"🚨 **PHISHING DETECTED** 🚨\n\n"
            message += f"🌐 **URL:** {result['url']}\n"
            message += f"⚠️ **Risk Level:** {result['risk_level']}\n"
            message += f"🎯 **Confidence:** {result['confidence'].title()}\n"
            message += f"📊 **Score:** {result['final_score']}/1.0\n\n"
            message += f"❌ **DO NOT VISIT THIS LINK** - It appears to be malicious!\n\n"
            message += f"🛡️ **Stay Safe:** Never enter personal information on suspicious sites."
        else:
            message = f"✅ **URL APPEARS SAFE** ✅\n\n"
            message += f"🌐 **URL:** {result['url']}\n"
            message += f"✅ **Risk Level:** {result['risk_level']}\n"
            message += f"🎯 **Confidence:** {result['confidence'].title()}\n"
            message += f"📊 **Score:** {result['final_score']}/1.0\n\n"
            message += f"✅ **This link appears legitimate**, but always exercise caution online.\n\n"
            message += f"💡 **Tip:** Always verify the URL matches the expected website."

        return {
            "success": True,
            "message": message,
            "data": result,
            "whatsapp_format": True
        }
        
    except Exception as e:
        logger.error(f"API analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/batch-analyze")
async def batch_analyze(request: BatchAnalyzeRequest):
    """Analyze multiple URLs"""
    try:
        if len(request.urls) > MAX_BATCH_SIZE:
            raise HTTPException(status_code=400, detail=f"Maximum {MAX_BATCH_SIZE} URLs allowed")
        
        results = []
        phishing_count = 0
        
        for url in request.urls:
            result = await analyze_url_core(url, request.user_id)
            results.append(result)
            if result.get("success") and result.get("is_phishing"):
                phishing_count += 1
        
        return {
            "success": True,
            "total_urls": len(request.urls),
            "phishing_detected": phishing_count,
            "safe_urls": len(request.urls) - phishing_count,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/{user_id}")
async def get_user_stats(user_id: str):
    """Get statistics for a user"""
    user_stats = server_stats["user_requests"].get(user_id)
    if not user_stats:
        return {"message": f"No statistics found for user: {user_id}"}
    return user_stats

@app.get("/health")
async def health_check():
    """Server health check"""
    return {
        "status": "healthy",
        "timestamp": _now(),
        "models": {
            "ml_model": model is not None,
            "vectorizer": vectorizer is not None,
            "groq_api": GROQ_API_KEY is not None
        },
        "stats": {
            "total_predictions": server_stats["total_predictions"],
            "phishing_detected": server_stats["phishing_detected"],
            "uptime": str(datetime.now() - server_stats["startup_time"])
        }
    }

# --- MCP Tools (same as before) ---
@mcp.tool
def validate() -> str:
    """Required validation tool for Puch AI"""
    return MY_NUMBER

@mcp.tool
async def analyze_url_mcp(
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
    url: Annotated[str, Field(description="URL to analyze for phishing")],
    lambda_override: Annotated[Optional[float], Field(description="Custom lambda value (0-1)")] = None,
) -> str:
    """Analyze a single URL for phishing threats (MCP version)"""
    result = await analyze_url_core(url, puch_user_id, lambda_override)
    
    if not result["success"]:
        return f"❌ Error: {result['error']}"
    
    if result["is_phishing"]:
        summary = f"🚨 **PHISHING DETECTED** 🚨\n"
        summary += f"URL: {result['url']}\n"
        summary += f"Risk Level: {result['risk_level']}\n"
        summary += f"Score: {result['final_score']}/1.0\n"
        summary += "⚠️ **DO NOT VISIT THIS LINK**"
    else:
        summary = f"✅ **URL APPEARS SAFE** ✅\n"
        summary += f"URL: {result['url']}\n"
        summary += f"Risk Level: {result['risk_level']}\n"
        summary += f"Score: {result['final_score']}/1.0"
    
    return summary

@mcp.tool
def get_stats_mcp(
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")]
) -> str:
    """Return statistics for a specific user (MCP version)"""
    user_stats = server_stats["user_requests"].get(puch_user_id)
    if not user_stats:
        return f"📊 No statistics found for user: {puch_user_id}"
    
    return f"📊 **USER STATISTICS** 📊\n{json.dumps(user_stats, indent=2)}"

# --- Server Runner ---
def run_mcp_server():
    """Run MCP server in background"""
    logger.info(f"🚀 Starting MCP Server on port {MCP_PORT}...")
    mcp.run(transport="stdio")

async def main():
    """Main function to run both servers"""
    logger.info("🚀 Starting Combined Phishing Detection Server...")
    logger.info(f"📡 FastAPI Server: http://{HOST}:{PORT}")
    logger.info(f"🔌 MCP Server: Available for MCP clients")
    logger.info("🌐 API Documentation: http://localhost:8000/docs")
    
    # Run FastAPI server
    config = uvicorn.Config(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())