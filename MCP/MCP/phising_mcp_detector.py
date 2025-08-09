import asyncio
import os
import requests
import pickle
import re
import logging
import json
from datetime import datetime
from typing import Annotated, Optional
from urllib.parse import urlparse
import validators
from dotenv import load_dotenv

from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from mcp import ErrorData, McpError
from mcp.types import TextContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import Field, BaseModel

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phishing_mcp_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Environment Configuration ---
TOKEN = "superrandomstring"  # Change this to something random!
MY_NUMBER = "917498124973"  # Your WhatsApp number for Puch AI validation

# Optional Settings
DEFAULT_LAMBDA = 0.6  # Weight for ML vs Grok (0.6 = 60% Grok, 40% ML)
MODEL_PATH = ""  # Path to your trained model
VECTORIZER_PATH = ""  # Path to your vectorizer

# Grok AI Settings (Optional - leave empty if you don't have Grok API)
GROK_API_URL = ""  # Example: "https://api.x.ai/v1/chat/completions"
GROK_API_KEY = ""  # Your Grok API key


# Validation
if not TOKEN or TOKEN == "phishing_detector_2024_super_secure_token_xyz123456789":
    print("⚠️  WARNING: Please change TOKEN to a secure random value!")

if not MY_NUMBER or MY_NUMBER == "+1234567890":
    print("⚠️  WARNING: Please set MY_NUMBER to your actual WhatsApp number!")

assert TOKEN, "TOKEN is required"
assert MY_NUMBER, "MY_NUMBER is required"

logger.info("✅ Configuration loaded successfully")

# Global variables for model
model = None
vectorizer = None
server_stats = {
    "startup_time": datetime.now(),
    "total_predictions": 0,
    "phishing_detected": 0,
    "errors": 0,
    "user_requests": {}  # Track requests per user
}

# --- Auth Provider (same pattern as your sample) ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    """
    A simple BearerAuthProvider that does not require any specific configuration.
    It allows any valid bearer token to access the MCP server.
    """

    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(
            public_key=k.public_key, jwks_uri=None, issuer=None, audience=None
        )
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="phishing-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Initialize MCP Server ---
mcp = FastMCP(
    "Phishing Detection MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

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

# Load models on startup
load_models()

# --- Rich Tool Description Models ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Utility Functions ---
def _now() -> str:
    return datetime.utcnow().isoformat()

def _error(code, msg):
    raise McpError(ErrorData(code=code, message=msg))

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
            
        # Try adding protocol if missing
        if not url.startswith(('http://', 'https://')):
            test_url = f"https://{url}"
            return validators.url(test_url)
            
        return False
    except:
        return False

def ml_model_score(url: str) -> tuple[float, bool]:
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

def grok_score(url: str, timeout: int = 10) -> tuple[float, bool]:
    """Get phishing probability from Groq API"""
    try:
        if not GROK_API_URL or not GROK_API_KEY:
            logger.warning("Groq API not configured, using fallback score")
            return 0.5, False
        
        # Groq uses OpenAI-compatible chat completions format
        payload = {
            "model": "llama-3.1-8b-instant",  # or another supported model
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
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            GROK_API_URL, 
            json=payload, 
            headers=headers, 
            timeout=timeout
        )
        response.raise_for_status()
        
        # Parse Groq's response format
        response_data = response.json()
        output_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "0.5").strip()
        
        # Extract numeric value from response
        score_match = re.search(r"0?\.\d+|[01]\.?\d*", output_text)
        if score_match:
            score = float(score_match.group())
            return max(0.0, min(1.0, score)), True
        else:
            return 0.5, False
            
    except requests.RequestException as e:
        logger.error(f"Groq API request error: {e}")
        return 0.5, False
    except Exception as e:
        logger.error(f"Groq API unexpected error: {e}")
        return 0.5, False

def update_user_stats(puch_user_id: str, is_phishing: bool):
    """Update statistics for user and global"""
    server_stats["total_predictions"] += 1
    if is_phishing:
        server_stats["phishing_detected"] += 1
    
    # Track per user
    if puch_user_id not in server_stats["user_requests"]:
        server_stats["user_requests"][puch_user_id] = {
            "total": 0,
            "phishing_found": 0,
            "first_request": _now(),
            "last_request": _now()
        }
    
    user_stats = server_stats["user_requests"][puch_user_id]
    user_stats["total"] += 1
    if is_phishing:
        user_stats["phishing_found"] += 1
    user_stats["last_request"] = _now()

# --- Tool Descriptions ---
VALIDATE_DESCRIPTION = RichToolDescription(
    description="Validate the MCP server and return the configured number.",
    use_when="Required by Puch AI for server validation.",
    side_effects=None,
)

ANALYZE_URL_DESCRIPTION = RichToolDescription(
    description="Analyze a URL for phishing threats using ML model and Grok AI.",
    use_when="When a user provides a URL to check for phishing, suspicious links, or wants to verify if a website is safe.",
    side_effects="Records the analysis in server statistics and user request history."
)

BATCH_ANALYZE_DESCRIPTION = RichToolDescription(
    description="Analyze multiple URLs at once for phishing threats (up to 10 URLs).",
    use_when="When a user provides multiple URLs to check simultaneously.",
    side_effects="Records all analyses in server statistics."
)

GET_STATS_DESCRIPTION = RichToolDescription(
    description="Get phishing detection statistics for a specific user.",
    use_when="When a user asks about their usage statistics, previous checks, or wants to see their phishing detection history.",
    side_effects=None,
)

# --- MCP Tools ---
@mcp.tool(description=VALIDATE_DESCRIPTION.model_dump_json())
async def validate() -> str:
    """Required validation tool for Puch AI"""
    return MY_NUMBER

@mcp.tool(description=ANALYZE_URL_DESCRIPTION.model_dump_json())
async def analyze_url(
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
    url: Annotated[str, Field(description="URL to analyze for phishing")],
    lambda_override: Annotated[Optional[float], Field(description="Custom lambda value (0-1) for ML/Grok weighting")] = None,
) -> list[TextContent]:
    """Analyze a single URL for phishing threats"""
    try:
        start_time = datetime.now()
        
        # Validate inputs
        if not puch_user_id:
            _error(INVALID_PARAMS, "puch_user_id is required")
        
        if not url or not url.strip():
            _error(INVALID_PARAMS, "URL cannot be empty")
        
        if not validate_url(url):
            _error(INVALID_PARAMS, "Invalid URL format")
        
        if lambda_override is not None and not (0 <= lambda_override <= 1):
            _error(INVALID_PARAMS, "Lambda must be between 0 and 1")
        
        # Use custom lambda if provided
        lambda_val = lambda_override if lambda_override is not None else DEFAULT_LAMBDA
        
        # Get predictions from both models
        ml_score, ml_success = ml_model_score(url)
        grok_score_val, grok_success = grok_score(url)
        
        # Calculate final score with weighted average
        if ml_success and grok_success:
            final_score = lambda_val * grok_score_val + (1 - lambda_val) * ml_score
        elif ml_success:
            final_score = ml_score
            logger.warning("Using ML score only (Grok unavailable)")
        elif grok_success:
            final_score = grok_score_val
            logger.warning("Using Grok score only (ML model unavailable)")
        else:
            final_score = 0.5
            logger.warning("Both models unavailable, using neutral score")
        
        # Ensure score is in valid range
        final_score = max(0.0, min(1.0, final_score))
        
        is_phishing = final_score > 0.5
        confidence = get_confidence_level(final_score)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update statistics
        update_user_stats(puch_user_id, is_phishing)
        
        # Log prediction
        logger.info(f"User {puch_user_id}: {url} -> {final_score:.3f} ({'PHISHING' if is_phishing else 'SAFE'})")
        
        # Create detailed response
        result = {
            "url": url,
            "analysis": {
                "is_phishing": is_phishing,
                "risk_level": "HIGH" if final_score >= 0.8 else "MEDIUM" if final_score >= 0.6 else "LOW",
                "confidence": confidence,
                "final_score": round(final_score, 4)
            },
            "model_scores": {
                "ml_model": round(ml_score, 4) if ml_success else "unavailable",
                "grok_ai": round(grok_score_val, 4) if grok_success else "unavailable",
                "lambda_weight": lambda_val
            },
            "metadata": {
                "processing_time_ms": round(processing_time, 2),
                "timestamp": _now(),
                "models_used": {
                    "ml_model": ml_success,
                    "grok_ai": grok_success
                }
            }
        }
        
        # Create user-friendly summary
        if is_phishing:
            summary = f"🚨 **PHISHING DETECTED** 🚨\n"
            summary += f"URL: {url}\n"
            summary += f"Risk Level: {result['analysis']['risk_level']}\n"
            summary += f"Confidence: {confidence.title()}\n"
            summary += f"Score: {final_score:.2f}/1.0\n\n"
            summary += "⚠️ **DO NOT VISIT THIS LINK** - It appears to be malicious!"
        else:
            summary = f"✅ **URL APPEARS SAFE** ✅\n"
            summary += f"URL: {url}\n"
            summary += f"Risk Level: {result['analysis']['risk_level']}\n"
            summary += f"Confidence: {confidence.title()}\n"
            summary += f"Score: {final_score:.2f}/1.0\n\n"
            summary += "This link appears to be legitimate, but always exercise caution online."
        
        return [
            TextContent(type="text", text=summary),
            TextContent(type="text", text=f"Detailed Analysis: {json.dumps(result, indent=2)}")
        ]
        
    except McpError:
        raise
    except Exception as e:
        server_stats["errors"] += 1
        logger.error(f"Analysis error for user {puch_user_id}, URL {url}: {e}")
        _error(INTERNAL_ERROR, f"Analysis failed: {str(e)}")

@mcp.tool(description=BATCH_ANALYZE_DESCRIPTION.model_dump_json())
async def batch_analyze_urls(
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
    urls: Annotated[list[str], Field(description="List of URLs to analyze (max 10)")],
) -> list[TextContent]:
    """Analyze multiple URLs for phishing threats"""
    try:
        if not puch_user_id:
            _error(INVALID_PARAMS, "puch_user_id is required")
        
        if not urls or len(urls) == 0:
            _error(INVALID_PARAMS, "URLs list cannot be empty")
        
        if len(urls) > 10:
            _error(INVALID_PARAMS, "Maximum 10 URLs allowed per batch")
        
        results = []
        phishing_count = 0
        
        for i, url in enumerate(urls, 1):
            try:
                if not validate_url(url):
                    results.append({
                        "url": url,
                        "error": "Invalid URL format",
                        "is_phishing": None
                    })
                    continue
                
                # Get predictions
                ml_score, ml_success = ml_model_score(url)
                grok_score_val, grok_success = grok_score(url)
                
                # Calculate final score
                if ml_success and grok_success:
                    final_score = DEFAULT_LAMBDA * grok_score_val + (1 - DEFAULT_LAMBDA) * ml_score
                elif ml_success:
                    final_score = ml_score
                elif grok_success:
                    final_score = grok_score_val
                else:
                    final_score = 0.5
                
                final_score = max(0.0, min(1.0, final_score))
                is_phishing = final_score > 0.5
                
                if is_phishing:
                    phishing_count += 1
                
                results.append({
                    "url": url,
                    "is_phishing": is_phishing,
                    "final_score": round(final_score, 4),
                    "risk_level": "HIGH" if final_score >= 0.8 else "MEDIUM" if final_score >= 0.6 else "LOW"
                })
                
                # Update stats
                update_user_stats(puch_user_id, is_phishing)
                
            except Exception as e:
                results.append({
                    "url": url,
                    "error": str(e),
                    "is_phishing": None
                })
        
        # Create summary
        total = len(urls)
        safe_count = sum(1 for r in results if r.get("is_phishing") is False)
        
        summary = f"📊 **BATCH ANALYSIS COMPLETE** 📊\n"
        summary += f"Total URLs: {total}\n"
        summary += f"🚨 Phishing Detected: {phishing_count}\n"
        summary += f"✅ Safe URLs: {safe_count}\n"
        summary += f"❌ Errors: {total - phishing_count - safe_count}\n\n"
        
        if phishing_count > 0:
            summary += "⚠️ **PHISHING URLS FOUND - AVOID THESE LINKS:**\n"
            for r in results:
                if r.get("is_phishing"):
                    summary += f"🚨 {r['url']} (Score: {r['final_score']})\n"
        
        return [
            TextContent(type="text", text=summary),
            TextContent(type="text", text=f"Detailed Results: {json.dumps(results, indent=2)}")
        ]
        
    except McpError:
        raise
    except Exception as e:
        server_stats["errors"] += 1
        logger.error(f"Batch analysis error for user {puch_user_id}: {e}")
        _error(INTERNAL_ERROR, f"Batch analysis failed: {str(e)}")

@mcp.tool(description=GET_STATS_DESCRIPTION.model_dump_json())
async def get_user_stats(
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
) -> list[TextContent]:
    """Get phishing detection statistics for a user"""
    try:
        if not puch_user_id:
            _error(INVALID_PARAMS, "puch_user_id is required")
        
        # Get user stats
        user_stats = server_stats["user_requests"].get(puch_user_id, {
            "total": 0,
            "phishing_found": 0,
            "first_request": "Never",
            "last_request": "Never"
        })
        
        # Calculate uptime
        uptime = (datetime.now() - server_stats["startup_time"]).total_seconds()
        uptime_hours = uptime / 3600
        
        # Create user-friendly summary
        if user_stats["total"] == 0:
            summary = f"📊 **YOUR PHISHING DETECTION STATS** 📊\n"
            summary += f"You haven't made any URL checks yet!\n"
            summary += f"Send me a URL to analyze and I'll check if it's safe or malicious."
        else:
            phishing_rate = (user_stats["phishing_found"] / user_stats["total"]) * 100
            
            summary = f"📊 **YOUR PHISHING DETECTION STATS** 📊\n"
            summary += f"Total URLs Checked: {user_stats['total']}\n"
            summary += f"🚨 Phishing Detected: {user_stats['phishing_found']}\n"
            summary += f"✅ Safe URLs: {user_stats['total'] - user_stats['phishing_found']}\n"
            summary += f"📈 Phishing Rate: {phishing_rate:.1f}%\n"
            summary += f"📅 First Check: {user_stats['first_request']}\n"
            summary += f"🕐 Last Check: {user_stats['last_request']}\n"
        
        # Server stats
        global_phishing_rate = (server_stats["phishing_detected"] / max(server_stats["total_predictions"], 1)) * 100
        
        detailed_stats = {
            "user_stats": user_stats,
            "server_stats": {
                "total_predictions": server_stats["total_predictions"],
                "phishing_detected": server_stats["phishing_detected"],
                "errors": server_stats["errors"],
                "uptime_hours": round(uptime_hours, 2),
                "global_phishing_rate": round(global_phishing_rate, 2)
            }
        }
        
        return [
            TextContent(type="text", text=summary),
            TextContent(type="text", text=f"Detailed Stats: {json.dumps(detailed_stats, indent=2)}")
        ]
        
    except McpError:
        raise
    except Exception as e:
        logger.error(f"Stats error for user {puch_user_id}: {e}")
        _error(INTERNAL_ERROR, f"Failed to get stats: {str(e)}")

# --- Main ---
async def main():
    try:
        logger.info("🚀 Starting Phishing Detection MCP Server...")
        logger.info(f"🤖 ML Model: {'Loaded' if model and vectorizer else 'Missing'}")
        logger.info(f"🧠 Grok API: {'Configured' if GROK_API_URL and GROK_API_KEY else 'Not Configured'}")
        logger.info(f"🌐 Server will run on: http://0.0.0.0:8086")
        logger.info("📡 To expose publicly, run: ngrok http 8086")
        logger.info("🔗 Then use the ngrok URL in Puch AI configuration")
        logger.info("=" * 60)
        
        await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)
        
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())