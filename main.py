from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os
from datetime import datetime, timezone
import json
import logging
from dotenv import load_dotenv
import asyncio
import re
import uuid

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="SEO Title Generator Agent",
    description="AI-powered SEO title generator for Telex.im",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Pydantic models
# ---------------------------
class SEOTitle(BaseModel):
    title: str
    description: str
    character_count: int
    keywords: List[str]

class AgentResponse(BaseModel):
    success: bool
    titles: List[SEOTitle]
    message: str
    timestamp: str

class GenerateRequest(BaseModel):
    topic: str
    keywords: Optional[str] = None

# ---------------------------
# Google Gemini initialization
# ---------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DISABLE_AI = os.getenv("DISABLE_AI", "false").lower() in ("1", "true", "yes")
USE_FAST_MODE = os.getenv("USE_FAST_MODE", "false").lower() in ("1", "true", "yes")
TEST_MODE = os.getenv("TEST_MODE", "false").lower() in ("1", "true", "yes")

model = None
if not DISABLE_AI and GEMINI_API_KEY and not TEST_MODE:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        preferred_models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro"
        ]
        for model_name in preferred_models:
            try:
                model = genai.GenerativeModel(model_name)
                logger.info(f"✅ Using model: {model_name}")
                break
            except Exception as e:
                logger.warning(f"⚠️ {model_name} unavailable: {str(e)}")
        if not model:
            logger.warning("❌ No Gemini models available, using fallback")
    except Exception as e:
        logger.error(f"❌ Gemini initialization failed: {str(e)}")
else:
    logger.info(f"🔄 Running in {'TEST' if TEST_MODE else 'FALLBACK'} mode")

# ---------------------------
# Fallback title generator
# ---------------------------
def generate_fallback_titles(topic: str, keywords: Optional[str] = None) -> List[SEOTitle]:
    if keywords and keywords.strip():
        keyword_list = [k.strip() for k in keywords.split(',') if k.strip()][:3]
    else:
        keyword_list = [topic.split()[0] if topic else "guide", "tips", "2025"]

    templates = [
        f"{topic.title()}: Complete Guide for 2025",
        f"Top 10 {topic.title()} Tips and Tricks",
        f"Master {topic.title()}: Expert Strategies",
        f"Ultimate {topic.title()} Guide You Need",
        f"{topic.title()} Explained: Simple Guide",
        f"Best {topic.title()} Practices That Work",
        f"{topic.title()}: Pro Tips Revealed",
        f"Boost Your {topic.title()} Results Now",
        f"{topic.title()} 101: Essential Guide",
        f"Advanced {topic.title()} Techniques"
    ]

    seo_titles = []
    for title_template in templates[:10]:
        title_text = title_template[:60]
        seo_titles.append(SEOTitle(
            title=title_text,
            description=f"Discover proven strategies and expert insights for {topic}. Learn actionable tips to achieve your goals.",
            character_count=len(title_text),
            keywords=keyword_list
        ))
    return seo_titles

# ---------------------------
# Extract JSON from AI response
# ---------------------------
def extract_json_from_text(text: str):
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except:
        pass
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return None

# ---------------------------
# Generate SEO titles
# ---------------------------
def generate_seo_titles_sync(topic: str, keywords: Optional[str] = None) -> List[SEOTitle]:
    if DISABLE_AI or not model or TEST_MODE or USE_FAST_MODE:
        return generate_fallback_titles(topic, keywords)

    keyword_hint = f"\nKeywords: {keywords}" if keywords else ""
    prompt = f"""Generate 10 SEO titles for: "{topic}"{keyword_hint}

Return ONLY a JSON array (no markdown, no explanation):
[{{"title":"Title Here","description":"Description here.","keywords":["kw1","kw2","kw3"]}}]

Rules:
- Exactly 10 titles
- Titles: 50-60 characters
- Descriptions: 140-160 characters
- 3 keywords per title
- Valid JSON only"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=1500,
                temperature=0.8,
                top_p=0.9,
                top_k=20
            )
        )
        content = response.text.strip()
        titles_data = extract_json_from_text(content)
        if not titles_data or len(titles_data) < 5:
            return generate_fallback_titles(topic, keywords)

        seo_titles = []
        for item in titles_data[:10]:
            seo_titles.append(SEOTitle(
                title=item.get("title", f"SEO Title for {topic}")[:60],
                description=item.get("description", f"Learn about {topic}")[:160],
                character_count=len(item.get("title", "")[:60]),
                keywords=item.get("keywords", [topic])[:3]
            ))
        while len(seo_titles) < 10:
            fallback = generate_fallback_titles(topic, keywords)
            seo_titles.extend(fallback[len(seo_titles):10])
        return seo_titles[:10]
    except Exception as e:
        logger.error(f"AI generation failed: {str(e)}")
        return generate_fallback_titles(topic, keywords)

# ---------------------------
# Format titles for Telex
# ---------------------------
def format_titles_for_telex(titles: List[SEOTitle]) -> str:
    message = "**Your 10 SEO-Optimized Titles:**\n\n"
    for i, title in enumerate(titles, 1):
        message += f"**{i}. {title.title}**\n"
        message += f"Description: {title.description}\n"
        message += f"Keywords: {', '.join(title.keywords)} | Length: {title.character_count} chars\n\n"
    message += "**Quick Tips:** Use numbers, power words, and keep under 60 chars for best CTR!"
    return message

# ---------------------------
# Extract message from Telex request
# ---------------------------
def extract_message_from_telex(body: dict) -> str:
    message_content = ""
    try:
        if "text" in body:
            message_content = body["text"]
        elif "message" in body:
            msg = body["message"]
            if isinstance(msg, str):
                message_content = msg
            elif isinstance(msg, dict):
                for field in ["text", "content"]:
                    if field in msg:
                        message_content = msg[field]
                        break
                if not message_content and "parts" in msg:
                    for part in msg["parts"]:
                        if part.get("kind") == "text":
                            message_content = part.get("text", "")
                            break
        elif "params" in body:
            params = body["params"]
            if "text" in params:
                message_content = params["text"]
            elif "message" in params:
                msg = params["message"]
                if isinstance(msg, str):
                    message_content = msg
                elif isinstance(msg, dict):
                    for field in ["text", "content"]:
                        if field in msg:
                            message_content = msg[field]
                            break
                    if not message_content and "parts" in msg:
                        for part in msg["parts"]:
                            if part.get("kind") == "text":
                                message_content = part.get("text", "")
                                break
        elif "content" in body:
            message_content = body["content"]
        elif "query" in body:
            message_content = body["query"]
    except Exception as e:
        logger.error(f"Error extracting message: {str(e)}")
    return message_content.strip() if message_content else ""

# ---------------------------
# Root endpoint
# ---------------------------
@app.get("/")
async def root():
    return {
        "status": "online",
        "agent": "SEO Title Generator",
        "version": "1.0.0",
        "mode": "test" if TEST_MODE else ("fast" if USE_FAST_MODE else "ai"),
        "ai_status": "connected" if model else "fallback",
        "endpoints": {
            "webhook": "/webhook",
            "health": "/health",
            "test": "/test"
        }
    }

# ---------------------------
# Health check
# ---------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "test" if TEST_MODE else ("fast" if USE_FAST_MODE else "ai"),
        "ai_enabled": bool(model)
    }

# ---------------------------
# Test endpoint
# ---------------------------
@app.post("/test")
async def test_webhook():
    return {
        "response": "✅ Webhook is working! Try: 'generate seo titles for: your topic'",
        "success": True
    }

# ---------------------------
# Telex Webhook endpoint
# ---------------------------
@app.post("/webhook")
async def telex_webhook(request: Request):
    rpc_id = "1"
    try:
        body = await request.json()
        rpc_id = body.get("id") or body.get("requestId") or rpc_id

        message_content = extract_message_from_telex(body)
        if not message_content:
            message_content = "Please provide a topic!\nExample: generate seo titles for: AI trends"

        prefixes = ['generate seo titles for:', 'seo titles for:', 'seo:', 'generate:', 'create:']
        topic_lower = message_content.lower()
        for prefix in prefixes:
            if topic_lower.startswith(prefix):
                message_content = message_content[len(prefix):].strip()
                break

        titles = await asyncio.to_thread(generate_seo_titles_sync, message_content)
        formatted_message = format_titles_for_telex(titles)

        response_payload = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "id": str(uuid.uuid4()),
                "contextId": "",
                "status": {
                    "state": "completed",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "message": {
                        "messageId": str(uuid.uuid4()),
                        "role": "agent",
                        "kind": "message",
                        "parts": [
                            {"kind": "text", "text": formatted_message}
                        ]
                    }
                },
                "kind": "task"
            }
        }
        return response_payload

    except Exception as e:
        logger.error(f"❌ Webhook error: {str(e)}", exc_info=True)
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {
                "code": -32000,
                "message": "Internal server error",
                "data": str(e)
            }
        }

# ---------------------------
# Direct API for testing
# ---------------------------
@app.post("/generate")
async def generate_titles_api(request: GenerateRequest):
    try:
        titles = generate_seo_titles_sync(request.topic, request.keywords)
        return AgentResponse(
            success=True,
            titles=titles,
            message=f"Generated {len(titles)} titles",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        logger.error(f"Generate API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    logger.info(f"🚀 Starting SEO Title Generator on {host}:{port}")
    logger.info(f"📊 Mode: {'TEST' if TEST_MODE else ('FAST' if USE_FAST_MODE else 'AI')}")
    logger.info(f"🤖 AI: {'Disabled' if DISABLE_AI else ('Connected' if model else 'No API Key')}")
    uvicorn.run(app, host=host, port=port)
