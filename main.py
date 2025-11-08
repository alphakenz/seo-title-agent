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

# CORS middleware - Configure for Telex
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Keep * for now, restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
USE_FAST_MODE = os.getenv("USE_FAST_MODE", "false").lower() in ("1", "true", "yes")
TEST_MODE = os.getenv("TEST_MODE", "false").lower() in ("1", "true", "yes")

# Initialize model with better error handling
model = None
if not DISABLE_AI and GEMINI_API_KEY and not TEST_MODE:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Use correct model names
        preferred_models = [
            'gemini-1.5-flash',  # Fast and efficient
            'gemini-1.5-pro',    # More capable
            'gemini-pro'         # Standard
        ]
        
        for model_name in preferred_models:
            try:
                model = genai.GenerativeModel(model_name)
                logger.info(f"✅ Using model: {model_name}")
                break
            except Exception as e:
                logger.warning(f"⚠️ {model_name} unavailable: {str(e)}")
                continue
                
        if not model:
            logger.warning("❌ No Gemini models available, using fallback")
    except Exception as e:
        logger.error(f"❌ Gemini initialization failed: {str(e)}")
else:
    logger.info(f"🔄 Running in {'TEST' if TEST_MODE else 'FALLBACK'} mode")

def generate_fallback_titles(topic: str, keywords: Optional[str] = None) -> List[SEOTitle]:
    """Generate instant fallback titles - FAST and reliable."""
    if keywords and keywords.strip():
        keyword_list = [k.strip() for k in keywords.split(',') if k.strip()][:3]
    else:
        keyword_list = [topic.split()[0] if topic else "guide", "tips", "2025"]
    
    # Smart title templates
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
    for i, title_template in enumerate(templates[:10]):
        title_text = title_template[:60]
        seo_titles.append(SEOTitle(
            title=title_text,
            description=f"Discover proven strategies and expert insights for {topic}. Learn actionable tips to achieve your goals.",
            character_count=len(title_text),
            keywords=keyword_list
        ))
    
    logger.info(f"✅ Generated {len(seo_titles)} fallback titles")
    return seo_titles

def extract_json_from_text(text: str):
    """Extract JSON array from AI response text."""
    # Remove markdown code blocks
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Try to parse as-is first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except:
        pass
    
    # Try to find JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    
    return None

def generate_seo_titles_sync(topic: str, keywords: Optional[str] = None) -> List[SEOTitle]:
    """Synchronous title generation."""
    
    # Use fallback if no model or AI is disabled
    if DISABLE_AI or not model or TEST_MODE:
        return generate_fallback_titles(topic, keywords)
    
    # Try AI generation with timeout handling in mind
    if USE_FAST_MODE:
        # In fast mode, go straight to fallback for speed
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
            logger.warning("Insufficient AI response, using fallback")
            return generate_fallback_titles(topic, keywords)
        
        # Convert to SEOTitle objects
        seo_titles = []
        for item in titles_data[:10]:
            try:
                seo_titles.append(SEOTitle(
                    title=item.get('title', f'SEO Title for {topic}')[:60],
                    description=item.get('description', f'Learn about {topic}')[:160],
                    character_count=len(item.get('title', '')[:60]),
                    keywords=item.get('keywords', [topic])[:3]
                ))
            except:
                continue
        
        # Ensure we have 10 titles
        while len(seo_titles) < 10:
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
        message += f"🏷️ {', '.join(title.keywords)} • {title.character_count} chars\n\n"
    
    message += "💡 **Quick Tips:** Use numbers, power words, and keep under 60 chars for best CTR!"
    
    return message

def extract_message_from_telex(body: dict) -> str:
    """Extract message content from various Telex request formats."""
    message_content = ""
    
    try:
        # Format 1: Standard webhook format with text field
        if "text" in body:
            message_content = body["text"]
        
        # Format 2: Message field with content
        elif "message" in body:
            if isinstance(body["message"], str):
                message_content = body["message"]
            elif isinstance(body["message"], dict):
                if "text" in body["message"]:
                    message_content = body["message"]["text"]
                elif "content" in body["message"]:
                    message_content = body["message"]["content"]
                elif "parts" in body["message"]:
                    for part in body["message"]["parts"]:
                        if part.get("kind") == "text":
                            message_content = part.get("text", "")
                            break
        
        # Format 3: Params structure (JSON-RPC style)
        elif "params" in body:
            params = body["params"]
            if "text" in params:
                message_content = params["text"]
            elif "message" in params:
                msg = params["message"]
                if isinstance(msg, str):
                    message_content = msg
                elif isinstance(msg, dict):
                    if "text" in msg:
                        message_content = msg["text"]
                    elif "content" in msg:
                        message_content = msg["content"]
                    elif "parts" in msg:
                        for part in msg["parts"]:
                            if part.get("kind") == "text":
                                message_content = part.get("text", "")
                                break
        
        # Format 4: Direct content field
        elif "content" in body:
            message_content = body["content"]
        
        # Format 5: Query field
        elif "query" in body:
            message_content = body["query"]
            
    except Exception as e:
        logger.error(f"Error extracting message: {str(e)}")
    
    return message_content.strip() if message_content else ""

@app.get("/")
async def root():
    """Root endpoint - agent info."""
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

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "test" if TEST_MODE else ("fast" if USE_FAST_MODE else "ai"),
        "ai_enabled": bool(model)
    }

@app.post("/test")
async def test_webhook():
    """Test endpoint that always works."""
    return {
        "response": "✅ Webhook is working! Try: 'generate seo titles for: your topic'",
        "success": True
    }

@app.post("/webhook")
async def telex_webhook(request: Request):
    """Main webhook endpoint for Telex.im integration."""
    
    start_time = datetime.now(timezone.utc)
    
    try:
        # Get request body
        body = await request.json()
        logger.info(f"📨 Webhook called at {start_time.isoformat()}")
        logger.info(f"📋 Request body keys: {list(body.keys())}")
        
        # Extract message
        message_content = extract_message_from_telex(body)
        logger.info(f"📝 Extracted message: {message_content[:100]}")
        
        # TEST MODE - return immediately
        if TEST_MODE:
            test_response = {
                "response": f"🧪 **TEST MODE**\n\nReceived: '{message_content}'\n\nWebhook is working! Disable TEST_MODE to generate real titles.",
                "success": True
            }
            logger.info("Returning TEST response")
            return test_response
        
        # Handle empty messages
        if not message_content:
            return {
                "response": "👋 **SEO Title Generator**\n\nSay: `generate seo titles for: [your topic]`",
                "success": True
            }
        
        # Help/info messages
        if message_content.lower() in ['help', 'info', 'how', '?', 'hello', 'hi']:
            help_msg = """👋 **SEO Title Generator**

🎯 I create 10 SEO-optimized titles instantly!

**How to use:**
• `generate seo titles for: [topic]`
• `seo: [topic]`

**Example:**
`generate seo titles for: sustainable fashion`"""
            
            return {
                "response": help_msg,
                "success": True
            }
        
        # Extract topic
        topic = message_content
        keywords = None
        
        # Handle "topic | keywords: word1, word2" format
        if '|' in message_content:
            parts = message_content.split('|', 1)
            topic = parts[0].strip()
            if len(parts) > 1 and 'keyword' in parts[1].lower():
                keywords = parts[1].split(':', 1)[-1].strip()
        
        # Clean topic - remove command prefixes
        prefixes = ['generate seo titles for:', 'seo titles for:', 'seo:', 'generate:', 'create:']
        topic_lower = topic.lower()
        for prefix in prefixes:
            if topic_lower.startswith(prefix):
                topic = topic[len(prefix):].strip()
                break
        
        # Validate topic
        if not topic or len(topic) < 2:
            return {
                "response": "❌ Please provide a topic!\n\nExample: `generate seo titles for: AI trends`",
                "success": True
            }
        
        logger.info(f"🎯 Generating for topic: {topic}")
        
        # Generate titles with timeout
        timeout_duration = 4.0 if USE_FAST_MODE else 7.0
        
        try:
            titles = await asyncio.wait_for(
                asyncio.to_thread(generate_seo_titles_sync, topic, keywords),
                timeout=timeout_duration
            )
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Timeout after {timeout_duration}s")
            titles = generate_fallback_titles(topic, keywords)
        
        # Format response
        formatted_message = format_titles_for_telex(titles)
        
        # Log timing
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(f"✅ Response ready in {duration:.2f}s")
        
        return {
            "response": formatted_message,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Webhook error: {str(e)}", exc_info=True)
        
        # Return user-friendly error
        return {
            "response": "❌ An error occurred. Please try again with a simpler topic.",
            "success": False
        }

@app.post("/generate")
async def generate_titles_api(request: GenerateRequest):
    """Direct API endpoint for testing."""
    
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"  # Must be 0.0.0.0 for Railway
    
    logger.info(f"🚀 Starting SEO Title Generator on {host}:{port}")
    logger.info(f"📊 Mode: {'TEST' if TEST_MODE else ('FAST' if USE_FAST_MODE else 'AI')}")
    logger.info(f"🤖 AI: {'Disabled' if DISABLE_AI else ('Connected' if model else 'No API Key')}")
    
    uvicorn.run(app, host=host, port=port)