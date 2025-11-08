from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import uuid
import asyncio
import re
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
import httpx
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="SEO Title Generator Agent",
    description="AI-powered SEO title generator for Telex.im",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic Models ----------
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

# ---------- Config ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DISABLE_AI = os.getenv("DISABLE_AI", "false").lower() in ("1","true","yes")
USE_FAST_MODE = os.getenv("USE_FAST_MODE", "false").lower() in ("1","true","yes")
TEST_MODE = os.getenv("TEST_MODE", "false").lower() in ("1","true","yes")

# Initialize Gemini AI
model = None
if not DISABLE_AI and GEMINI_API_KEY and not TEST_MODE:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        preferred_models = ['gemini-1.5-flash','gemini-1.5-pro','gemini-pro']
        for name in preferred_models:
            try:
                model = genai.GenerativeModel(name)
                logger.info(f"✅ Using model: {name}")
                break
            except Exception as e:
                logger.warning(f"⚠️ {name} unavailable: {e}")
        if not model:
            logger.warning("❌ No Gemini model available, fallback mode")
    except Exception as e:
        logger.error(f"❌ Gemini initialization failed: {e}")
else:
    logger.info(f"🔄 Running in {'TEST' if TEST_MODE else 'FALLBACK'} mode")

# ---------- Fallback ----------
def generate_fallback_titles(topic: str, keywords: Optional[str] = None) -> List[SEOTitle]:
    kw_list = [k.strip() for k in keywords.split(',')[:3]] if keywords else [topic.split()[0] if topic else "guide", "tips", "2025"]
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
    titles = []
    for t in templates[:10]:
        title_text = t[:60]
        titles.append(SEOTitle(
            title=title_text,
            description=f"Discover proven strategies and expert insights for {topic}. Learn actionable tips to achieve your goals.",
            character_count=len(title_text),
            keywords=kw_list
        ))
    return titles

# ---------- AI SEO Generation ----------
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

def generate_seo_titles_sync(topic: str, keywords: Optional[str] = None) -> List[SEOTitle]:
    if DISABLE_AI or not model or TEST_MODE or USE_FAST_MODE:
        return generate_fallback_titles(topic, keywords)
    keyword_hint = f"\nKeywords: {keywords}" if keywords else ""
    prompt = f"""Generate 10 SEO titles for: "{topic}"{keyword_hint}
Return ONLY JSON array:
[{{"title":"Title","description":"Description","keywords":["kw1","kw2","kw3"]}}]
Rules: 10 titles, 50-60 chars, 140-160 desc, 3 keywords"""
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
        data = extract_json_from_text(response.text) or []
        seo_titles = []
        for item in data[:10]:
            seo_titles.append(SEOTitle(
                title=item.get('title', '')[:60],
                description=item.get('description', '')[:160],
                character_count=len(item.get('title', '')[:60]),
                keywords=item.get('keywords', [topic])[:3]
            ))
        while len(seo_titles) < 10:
            seo_titles.extend(generate_fallback_titles(topic, keywords)[len(seo_titles):])
        return seo_titles[:10]
    except Exception as e:
        logger.error(f"AI generation failed: {e}")
        return generate_fallback_titles(topic, keywords)

def format_titles_for_telex(titles: List[SEOTitle]) -> str:
    msg = "**Your 10 SEO-Optimized Titles:**\n\n"
    for i, t in enumerate(titles,1):
        msg += f"**{i}. {t.title}**\nDescription: {t.description}\nKeywords: {', '.join(t.keywords)} | Length: {t.character_count} chars\n\n"
    msg += "**Quick Tips:** Use numbers, power words, and keep under 60 chars for best CTR!"
    return msg

# ---------- Async push ----------
async def push_to_telex(push_url: str, token: str, message: dict):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(push_url, headers=headers, json=message)

async def generate_and_push_for_topic(topic: str, push_url: str, token: str):
    titles = await asyncio.to_thread(generate_seo_titles_sync, topic)
    msg = format_titles_for_telex(titles)
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "kind": "message",
                "role": "agent",
                "parts": [{"kind": "text", "text": msg}],
                "metadata": {},
                "messageId": str(uuid.uuid4()),
                "contextId": ""
            }
        }
    }
    await push_to_telex(push_url, token, payload)

# ---------- Telex Webhook ----------
@app.post("/webhook")
async def telex_webhook(request: Request):
    try:
        body = await request.json()
        rpc_id = body.get("id") or body.get("requestId") or "1"

        push_conf = body.get("params", {}).get("configuration", {}).get("pushNotificationConfig", {})
        push_url = push_conf.get("url")
        push_token = push_conf.get("token")
        if not push_url or not push_token:
            raise ValueError("Missing pushNotificationConfig url or token")

        # Extract topics
        message = body.get("params", {}).get("message", {})
        parts = message.get("parts", [])
        topics = [p.get("text").strip() for p in parts if p.get("kind")=="text" and p.get("text")]

        # Push async
        for t in topics:
            if t:
                asyncio.create_task(generate_and_push_for_topic(t, push_url, push_token))

        # Immediate response
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "id": str(uuid.uuid4()),
                "status": {"state": "running", "timestamp": datetime.utcnow().isoformat() + "Z"},
                "kind": "task"
            }
        }
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {
            "jsonrpc": "2.0",
            "id": body.get("id", "1"),
            "error": {"code": -32000, "message": str(e)}
        }

# ---------- Direct API ----------
@app.post("/generate")
async def generate_titles_api(req: GenerateRequest):
    try:
        titles = generate_seo_titles_sync(req.topic, req.keywords)
        return AgentResponse(
            success=True,
            titles=titles,
            message=f"Generated {len(titles)} titles",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        logger.error(f"Generate API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Root & Health ----------
@app.get("/")
async def root():
    return {"status":"online","agent":"SEO Title Generator","version":"1.0.0"}

@app.get("/health")
async def health_check():
    return {"status":"healthy","timestamp":datetime.now(timezone.utc).isoformat(),"ai_enabled":bool(model)}

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    logger.info(f"🚀 Starting SEO Title Generator on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
