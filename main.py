from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os
from datetime import datetime
import json
import logging
from dotenv import load_dotenv
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SEO Title Generator Agent",
    description="AI-powered SEO title generator for Telex.im",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=2)

# Pydantic models
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

# Initialize Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DISABLE_AI = os.getenv("DISABLE_AI", "false").lower() in ("1", "true", "yes")
USE_FAST_MODE = os.getenv("USE_FAST_MODE", "true").lower() in ("1", "true", "yes")

# Initialize model with better error handling
model = None
if not DISABLE_AI and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Try models in order of preference
        preferred_models = [
            'gemini-2.0-flash-exp',
            'gemini-1.5-flash',
            'gemini-1.5-flash-latest',
            'gemini-pro'
        ]
        
        for model_name in preferred_models:
            try:
                test_model = genai.GenerativeModel(model_name)
                test_response = test_model.generate_content(
                    "OK",
                    generation_config=genai.GenerationConfig(max_output_tokens=5)
                )
                model = test_model
                logger.info(f"✅ Using model: {model_name}")
                break
            except Exception as e:
                logger.warning(f"⚠️ {model_name} unavailable")
                continue
                
        if not model:
            logger.warning("❌ No Gemini models available, using fallback")
    except Exception as e:
        logger.error(f"❌ Gemini initialization failed: {str(e)}")

def generate_fallback_titles(topic: str, keywords: Optional[str] = None) -> List[SEOTitle]:
    """Generate instant fallback titles - FAST and reliable."""
    keyword_list = [k.strip() for k in keywords.split(',')] if keywords else [topic, "guide", "tips"]
    
    # Smart title templates
    templates = [
        f"{topic.title()}: Complete Guide for 2025",
        f"Top 10 {topic.title()} Tips and Tricks",
        f"Master {topic.title()}: Expert Strategies",
        f"Ultimate {topic.title()} Guide",
        f"{topic.title()} Explained: Simple Guide",
        f"Best {topic.title()} Practices That Work",
        f"{topic.title()}: Pro Tips Revealed",
        f"Boost Your {topic.title()} Results Now",
        f"{topic.title()} 101: Essential Guide",
        f"Advanced {topic.title()} Techniques"
    ]
    
    seo_titles = []
    for i, title_template in enumerate(templates):
        title_text = title_template[:60]
        seo_titles.append(SEOTitle(
            title=title_text,
            description=f"Discover proven strategies and expert insights for {topic}. Learn actionable tips and best practices to achieve your goals with this comprehensive guide.",
            character_count=len(title_text),
            keywords=keyword_list[:3] if keyword_list else [topic, "guide", "tips"]
        ))
    
    logger.info(f"✅ Generated {len(seo_titles)} fallback titles instantly")
    return seo_titles

def generate_seo_titles_sync(topic: str, keywords: Optional[str] = None) -> List[SEOTitle]:
    """Synchronous title generation - can be called in thread."""
    
    # CRITICAL: Use fallback if no model or fast mode enabled
    if DISABLE_AI or not model or USE_FAST_MODE:
        return generate_fallback_titles(topic, keywords)
    
    keyword_hint = f"\nKeywords: {keywords}" if keywords else ""
    
    prompt = f"""Generate 10 SEO titles for: "{topic}"{keyword_hint}

Return ONLY a JSON array (no markdown):
[{{"title":"Title Here","description":"Description here.","keywords":["kw1","kw2","kw3"]}}]

Rules:
- 10 titles, each 50-60 chars
- Descriptions: 140-160 chars
- 3 keywords each
- Pure JSON, no markdown"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=1200,
                temperature=0.7,
                top_p=0.8,
                top_k=15
            )
        )
        
        content = response.text.strip()
        
        # Clean response
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        content = content.replace('`', '')
        
        # Extract JSON
        start_idx = content.find('[')
        end_idx = content.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            logger.warning("No JSON array found, using fallback")
            return generate_fallback_titles(topic, keywords)
        
        json_content = content[start_idx:end_idx + 1]
        
        try:
            titles_data = json.loads(json_content)
        except json.JSONDecodeError:
            # Try to fix JSON
            json_content = json_content.replace("'", '"')
            json_content = re.sub(r',\s*}', '}', json_content)
            json_content = re.sub(r',\s*]', ']', json_content)
            try:
                titles_data = json.loads(json_content)
            except:
                logger.warning("JSON parsing failed, using fallback")
                return generate_fallback_titles(topic, keywords)
        
        if not isinstance(titles_data, list) or len(titles_data) < 5:
            return generate_fallback_titles(topic, keywords)
        
        # Convert to SEOTitle objects
        seo_titles = []
        for item in titles_data[:10]:
            try:
                seo_titles.append(SEOTitle(
                    title=item.get('title', 'SEO Title')[:60],
                    description=item.get('description', 'Description.')[:160],
                    character_count=len(item.get('title', 'SEO Title')[:60]),
                    keywords=item.get('keywords', [topic])[:3]
                ))
            except:
                continue
        
        # Pad if needed
        if len(seo_titles) < 10:
            fallback = generate_fallback_titles(topic, keywords)
            seo_titles.extend(fallback[len(seo_titles):10])
        
        return seo_titles[:10]
        
    except Exception as e:
        logger.error(f"AI generation failed: {str(e)}")
        return generate_fallback_titles(topic, keywords)

def format_titles_for_telex(titles: List[SEOTitle]) -> str:
    """Format titles for Telex chat display."""
    
    message = "✨ **Your 10 SEO-Optimized Titles:**\n\n"
    
    for i, title in enumerate(titles, 1):
        message += f"**{i}. {title.title}**\n"
        message += f"📝 {title.description}\n"
        message += f"🏷️ {', '.join(title.keywords[:3])} • {title.character_count} chars\n\n"
    
    message += "💡 **SEO Tips:**\n"
    message += "• Test variations for best CTR\n"
    message += "• Use numbers & power words\n"
    message += "• Keep under 60 characters\n"
    
    return message

@app.get("/")
async def root():
    """Root endpoint - agent info."""
    return {
        "status": "online",
        "agent": "SEO Title Generator",
        "version": "1.0.0",
        "description": "AI-powered SEO title generator for Telex.im",
        "mode": "fast_fallback" if (DISABLE_AI or USE_FAST_MODE) else "ai_powered",
        "ai_status": "connected" if model else "fallback_mode",
        "endpoints": {
            "health": "/health",
            "webhook": "/webhook (POST)",
            "generate": "/generate (POST)",
            "info": "/agent-info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ai_model": "Google Gemini" if model else "Fallback",
        "mode": "fast" if USE_FAST_MODE else "standard",
        "version": "1.0.0"
    }

@app.get("/webhook")
async def webhook_health():
    """Webhook GET endpoint - for Telex verification."""
    return {
        "status": "ready",
        "message": "Webhook is active",
        "agent": "SEO Title Generator",
        "version": "1.0.0"
    }

@app.post("/webhook")
async def telex_webhook(request: Request):
    """
    Main webhook endpoint for Telex.im integration.
    
    This is called by Telex when users send messages to the agent.
    MUST respond within 10 seconds or Telex will timeout.
    """
    
    try:
        # Parse request body
        body = await request.json()
        logger.info(f"📨 Webhook called")
        
        # Extract message content from various Telex formats
        message_content = ""
        
        # Format 1: JSON-RPC with params.message.parts
        if "params" in body and "message" in body["params"]:
            message_obj = body["params"]["message"]
            if "parts" in message_obj:
                for part in message_obj["parts"]:
                    if part.get("kind") == "text":
                        message_content = part.get("text", "")
                        break
            elif "content" in message_obj:
                message_content = message_obj.get("content", "")
        
        # Format 2: Direct message object
        elif "message" in body:
            if isinstance(body["message"], dict):
                if "parts" in body["message"]:
                    for part in body["message"]["parts"]:
                        if part.get("kind") == "text":
                            message_content = part.get("text", "")
                            break
                elif "content" in body["message"]:
                    message_content = body["message"].get("content", "")
                elif "text" in body["message"]:
                    message_content = body["message"].get("text", "")
            elif isinstance(body["message"], str):
                message_content = body["message"]
        
        # Format 3: Direct content/text fields
        elif "content" in body:
            message_content = body.get("content", "")
        elif "text" in body:
            message_content = body.get("text", "")
        
        message_content = message_content.strip()
        logger.info(f"📝 Message: {message_content[:100]}")
        
        # Help message for non-SEO queries
        if not any(word in message_content.lower() for word in ['seo', 'title', 'generate', 'create']):
            help_msg = """👋 **SEO Title Generator**

🎯 I create 10 SEO-optimized titles instantly!

**Usage:**
• `generate seo titles for: [topic]`
• `seo: [topic] | keywords: [words]`

**Example:**
`generate seo titles for: sustainable fashion`

Ready to create amazing titles? 🚀"""
            
            return _format_response(body, help_msg)
        
        # Extract topic and keywords
        topic = message_content
        keywords = None
        
        # Handle "topic | keywords: word1, word2" format
        if '|' in message_content:
            parts = message_content.split('|', 1)
            topic = parts[0].strip()
            if len(parts) > 1 and 'keywords:' in parts[1].lower():
                keywords = parts[1].split('keywords:', 1)[1].strip()
        
        # Clean topic (remove command prefixes)
        for prefix in ['generate seo titles for:', 'seo titles for:', 'create seo titles for:',
                       'generate titles for:', 'create titles for:', 'seo:', 'generate:', 'create:']:
            if topic.lower().startswith(prefix):
                topic = topic[len(prefix):].strip()
                break
        
        # Validate topic
        if not topic or len(topic) < 2:
            error_msg = "❌ Please provide a valid topic!\n\nExample: `generate seo titles for: AI trends`"
            return _format_response(body, error_msg)
        
        logger.info(f"🎯 Topic: {topic}")
        if keywords:
            logger.info(f"🏷️ Keywords: {keywords}")
        
        # CRITICAL: Generate titles with STRICT timeout
        # Telex requires response within ~10 seconds
        try:
            titles = await asyncio.wait_for(
                asyncio.to_thread(generate_seo_titles_sync, topic, keywords),
                timeout=6.0  # 6 seconds max - leaves buffer for response formatting
            )
            logger.info(f"✅ Generated {len(titles)} titles")
        except asyncio.TimeoutError:
            logger.warning("⏱️ Timeout - using instant fallback")
            titles = generate_fallback_titles(topic, keywords)
        except Exception as e:
            logger.error(f"❌ Error: {str(e)}")
            titles = generate_fallback_titles(topic, keywords)
        
        # Format response
        formatted_message = format_titles_for_telex(titles)
        
        return _format_response(body, formatted_message, {
            "titles_generated": len(titles),
            "topic": topic,
            "keywords": keywords,
            "mode": "ai" if (model and not USE_FAST_MODE) else "fallback",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        
    except Exception as e:
        logger.error(f"❌ Webhook error: {str(e)}", exc_info=True)
        error_msg = "❌ Error occurred. Please try a simpler topic."
        
        try:
            return _format_response(body, error_msg)
        except:
            return {"response": error_msg}

def _format_response(body: dict, message: str, metadata: dict = None):
    """Format response based on request type (JSON-RPC or simple)."""
    
    # JSON-RPC format
    if "jsonrpc" in body and "id" in body:
        response = {
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {"response": message}
        }
        if metadata:
            response["result"]["metadata"] = metadata
        return response
    
    # Simple format
    response = {"response": message}
    if metadata:
        response["metadata"] = metadata
    return response

@app.post("/generate")
async def generate_titles_api(request: GenerateRequest):
    """Direct API endpoint for generating titles (non-Telex)."""
    
    try:
        titles = await asyncio.to_thread(
            generate_seo_titles_sync, 
            request.topic, 
            request.keywords
        )
        
        return AgentResponse(
            success=True,
            titles=titles,
            message=f"Generated {len(titles)} titles",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"Generate API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent-info")
async def agent_info():
    """Agent information for Telex.im discovery."""
    return {
        "name": "SEO Title Generator",
        "description": "Generate 10 SEO-optimized titles with descriptions and keywords",
        "version": "1.0.0",
        "author": "Built for HNG Internship",
        "ai_model": "Google Gemini (with instant fallback)",
        "mode": "fast_reliable" if USE_FAST_MODE else "ai_powered",
        "capabilities": [
            "Generate SEO-optimized titles (50-60 chars)",
            "Create meta descriptions (140-160 chars)",
            "Extract relevant keywords",
            "Optimize for CTR and search ranking"
        ],
        "usage": {
            "commands": [
                "generate seo titles for: [topic]",
                "seo: [topic] | keywords: [keywords]"
            ],
            "examples": [
                "generate seo titles for: AI in healthcare",
                "seo: Python tutorials | keywords: beginner, code, learn"
            ]
        },
        "endpoints": {
            "webhook": "/webhook (POST)",
            "health": "/health (GET)",
            "generate": "/generate (POST)",
            "info": "/agent-info (GET)"
        },
        "protocol": "Telex Webhook (REST)",
        "response_time": "< 10 seconds (guaranteed)"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 SEO Title Generator starting on {host}:{port}")
    logger.info(f"📊 Mode: {'Fast Fallback' if USE_FAST_MODE else 'AI Powered'}")
    uvicorn.run(app, host=host, port=port)