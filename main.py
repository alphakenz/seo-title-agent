from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
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

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SEO Title Generator Agent")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for Telex A2A protocol
class TelexMessage(BaseModel):
    id: str
    content: str
    sender: dict
    channel_id: str
    timestamp: Optional[str] = None

class TelexRequest(BaseModel):
    message: TelexMessage
    context: Optional[dict] = None

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

# Initialize Google Gemini (FREE API)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set. Agent will not function properly.")
    logger.warning("Please set GEMINI_API_KEY in .env file or environment variable")
    logger.warning("Get FREE API key at: https://makersuite.google.com/app/apikey")
    model = None
else:
    logger.info(f"✅ GEMINI_API_KEY loaded successfully (starts with: {GEMINI_API_KEY[:10]}...)")
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Prioritize Flash models (higher free tier limits)
    preferred_models = [
        'models/gemini-2.5-flash',
        'models/gemini-2.0-flash',
        'models/gemini-flash-latest',
        'models/gemini-2.5-flash-lite',
        'models/gemini-2.0-flash-lite'
    ]
    
    model = None
    working_model_name = None
    
    # Try preferred models first
    for model_name in preferred_models:
        try:
            logger.info(f"🔍 Trying model: {model_name}")
            test_model = genai.GenerativeModel(model_name)
            test_response = test_model.generate_content("Say OK", 
                generation_config=genai.GenerationConfig(max_output_tokens=10))
            logger.info(f"✅ Model {model_name} working!")
            model = test_model
            working_model_name = model_name
            break
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                logger.warning(f"⚠️ {model_name} quota exceeded, trying next...")
            else:
                logger.warning(f"⚠️ {model_name} failed: {str(e)[:100]}")
            continue
    
    if model:
        logger.info(f"🎉 Successfully initialized: {working_model_name}")
    else:
        logger.error("❌ All preferred models failed. You may need to wait for quota reset.")
        logger.info("💡 Free tier quotas reset daily. Try again in a few hours.")

def generate_fallback_titles(topic: str, keywords: Optional[str] = None) -> List[SEOTitle]:
    """Generate simple fallback titles if AI parsing fails."""
    keyword_list = [k.strip() for k in keywords.split(',')] if keywords else [topic, "guide", "tips"]
    
    base_titles = [
        f"{topic.title()}: Complete Guide for 2025",
        f"Top 10 {topic.title()} Tips and Tricks",
        f"Master {topic.title()}: Beginner to Expert",
        f"Ultimate {topic.title()} Guide You Need",
        f"{topic.title()} Explained Simply",
        f"Best {topic.title()} Strategies That Work",
        f"{topic.title()}: Expert Tips Revealed",
        f"Boost Your {topic.title()} Results Fast",
        f"{topic.title()} 101: Essential Guide",
        f"Advanced {topic.title()} Techniques"
    ]
    
    seo_titles = []
    for i, title in enumerate(base_titles):
        seo_titles.append(SEOTitle(
            title=title[:60],
            description=f"Discover proven strategies and expert insights for {topic}. Get started today and see results!",
            character_count=len(title[:60]),
            keywords=keyword_list[:3]
        ))
    
    logger.info(f"✅ Generated {len(seo_titles)} fallback titles")
    return seo_titles

def generate_seo_titles(topic: str, keywords: Optional[str] = None) -> List[SEOTitle]:
    """Generate 10 SEO-optimized titles with descriptions using Google Gemini (FREE)."""
    
    if not model:
        raise HTTPException(status_code=500, detail="AI model not initialized. Please set GEMINI_API_KEY")
    
    # Simplified prompt for better JSON compliance
    prompt = f"""Generate 10 SEO titles for: {topic}

Return ONLY this JSON format with NO other text:
[{{"title":"Example Title Here","description":"Example description here that is around 150 characters long.","keywords":["keyword1","keyword2","keyword3"]}},{{"title":"Another Title","description":"Another description here.","keywords":["word1","word2","word3"]}}]

Rules:
- Exactly 10 items
- Titles: 50-60 characters
- Descriptions: 150 characters
- 3 keywords each
- Pure JSON only - no markdown, no text before/after"""

    try:
        logger.info(f"Generating titles for topic: {topic}")
        
        # Optimize generation settings for speed and better JSON
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=3000,
                temperature=0.5,  # Lower for more consistent JSON
                top_p=0.8,
                top_k=20
            )
        )
        
        content = response.text
        logger.info("✅ Received response from Gemini API")
        
        # Aggressive JSON cleaning
        content = content.strip()
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        content = content.replace('`', '')
        
        # Find JSON array boundaries
        start_idx = content.find('[')
        end_idx = content.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            logger.error(f"No valid JSON array found in response: {content[:200]}")
            logger.info("Using fallback titles instead")
            return generate_fallback_titles(topic, keywords)
        
        json_content = content[start_idx:end_idx + 1]
        
        # Try to parse JSON
        try:
            titles_data = json.loads(json_content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Position: {e.pos}, Line: {e.lineno}, Column: {e.colno}")
            
            # Try aggressive fixes
            json_content = json_content.replace("'", '"')
            json_content = re.sub(r',\s*}', '}', json_content)  # Remove trailing commas
            json_content = re.sub(r',\s*]', ']', json_content)  # Remove trailing commas
            
            try:
                titles_data = json.loads(json_content)
                logger.info("✅ Fixed JSON successfully")
            except json.JSONDecodeError:
                logger.error("Could not fix JSON, using fallback")
                return generate_fallback_titles(topic, keywords)
        
        # Validate we have a list
        if not isinstance(titles_data, list):
            logger.error(f"Response is not a list: {type(titles_data)}")
            return generate_fallback_titles(topic, keywords)
        
        # Convert to SEOTitle objects
        seo_titles = []
        for item in titles_data[:10]:
            try:
                seo_titles.append(SEOTitle(
                    title=item.get('title', 'SEO Title')[:60],
                    description=item.get('description', 'Description here.')[:160],
                    character_count=len(item.get('title', '')[:60]),
                    keywords=item.get('keywords', [topic, 'guide', 'tips'])[:3]
                ))
            except Exception as e:
                logger.warning(f"Skipping invalid title: {str(e)}")
                continue
        
        # Ensure we have at least some titles
        if len(seo_titles) < 5:
            logger.warning(f"Only got {len(seo_titles)} titles, using fallback")
            return generate_fallback_titles(topic, keywords)
        
        # Pad with fallback if needed
        if len(seo_titles) < 10:
            fallback = generate_fallback_titles(topic, keywords)
            seo_titles.extend(fallback[len(seo_titles):10])
        
        logger.info(f"✅ Successfully generated {len(seo_titles)} titles")
        return seo_titles[:10]
        
    except Exception as e:
        logger.error(f"Error generating titles: {str(e)}")
        logger.info("Returning fallback titles")
        return generate_fallback_titles(topic, keywords)

def format_titles_for_telex(titles: List[SEOTitle]) -> str:
    """Format SEO titles for display in Telex chat."""
    
    message = "✅ **Your 10 SEO-Optimized Titles:**\n\n"
    
    for i, title in enumerate(titles, 1):
        message += f"**{i}. {title.title}**\n"
        message += f"📝 {title.description}\n"
        message += f"🏷️ {', '.join(title.keywords[:3])} • {title.character_count} chars\n\n"
    
    message += "💡 **Tips:** Test multiple titles • Optimize for CTR • A/B test results\n"
    
    return message

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "agent": "SEO Title Generator",
        "version": "1.0.0",
        "description": "AI agent that generates 10 SEO-optimized titles with descriptions",
        "ai_model": "Google Gemini 2.5 Flash (FREE)",
        "api_configured": model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ai_model": "Google Gemini 2.5 Flash (FREE)",
        "ai_client": "connected" if model else "disconnected"
    }

@app.get("/webhook")
async def webhook_health():
    """Health check endpoint for webhook - Telex.im checks this first."""
    return {
        "status": "ready",
        "message": "Webhook endpoint is active and ready to receive POST requests",
        "endpoint": "/webhook",
        "method": "POST",
        "agent": "SEO Title Generator",
        "version": "1.0.0"
    }

@app.post("/webhook")
async def telex_webhook(request: Request):
    """Main webhook endpoint for Telex.im A2A protocol."""
    
    try:
        # Log raw request for debugging
        raw_body = await request.body()
        logger.info(f"📨 Received webhook request")
        
        # Try to parse as TelexRequest
        try:
            body_json = await request.json()
            
            # Handle different request formats
            if "message" in body_json:
                telex_request = TelexRequest(**body_json)
                message_content = telex_request.message.content.strip()
                channel_id = telex_request.message.channel_id
            elif "content" in body_json:
                # Alternative format
                message_content = body_json.get("content", "").strip()
                channel_id = body_json.get("channel_id", "unknown")
            else:
                logger.error(f"❌ Unknown request format")
                return {
                    "response": "❌ Invalid request format"
                }
                
        except Exception as parse_error:
            logger.error(f"❌ Failed to parse request: {str(parse_error)}")
            return {
                "response": f"❌ Error parsing request"
            }
        
        logger.info(f"📝 Processing: {message_content[:50]}")
        
        # Parse the message to extract topic and keywords
        if not any(trigger in message_content.lower() for trigger in ['seo', 'title', 'generate']):
            return {
                "response": "👋 **Welcome! I'm SEO Muse, your AI SEO expert.**\n\n"
                           "🎯 I create SEO-optimized titles that rank!\n\n"
                           "**How to use:**\n"
                           "• `generate seo titles for: your topic`\n"
                           "• `seo: topic | keywords: word1, word2`\n\n"
                           "**Example:**\n"
                           "`generate seo titles for: WordPress SEO`\n\n"
                           "Ready to boost your SEO? 🚀"
            }
        
        # Extract topic and keywords
        topic = message_content
        keywords = None
        
        if '|' in message_content:
            parts = message_content.split('|')
            topic = parts[0].strip()
            if 'keywords:' in parts[1].lower():
                keywords = parts[1].split('keywords:', 1)[1].strip()
        
        # Clean up topic
        for prefix in ['generate seo titles for:', 'seo titles for:', 'seo:', 'generate titles for:']:
            if prefix in topic.lower():
                topic = topic.lower().replace(prefix, '').strip()
                break
        
        if not topic or len(topic) < 3:
            return {
                "response": "❌ Please provide a valid topic.\n\n"
                           "Example: `generate seo titles for: AI trends`"
            }
        
        # Send immediate acknowledgment with a professional processing message
        logger.info(f"🎯 Generating for: {topic}")
        
        # Simpler processing message for faster response
        processing_msg = f"🎨 Generating SEO titles for: **{topic}**\n\n"
        
        # Generate SEO titles with timeout protection
        try:
            titles = await asyncio.wait_for(
                asyncio.to_thread(generate_seo_titles, topic, keywords),
                timeout=25.0  # 25 second timeout
            )
        except asyncio.TimeoutError:
            logger.error("⏱️ Generation timed out")
            return {
                "response": "⏱️ **Request timed out**\n\n"
                           "Try: `generate seo titles for: simpler topic`"
            }
        
        # Format response for Telex
        formatted_message = processing_msg + format_titles_for_telex(titles)
        
        return {
            "response": formatted_message,
            "metadata": {
                "titles_generated": len(titles),
                "topic": topic,
                "ai_model": "Google Gemini 2.5 Flash (FREE)",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return {
            "response": "❌ An error occurred. Please try again."
        }

@app.post("/generate")
async def generate_titles_api(topic: str, keywords: Optional[str] = None):
    """Direct API endpoint for generating SEO titles (for testing)."""
    
    try:
        titles = generate_seo_titles(topic, keywords)
        
        return AgentResponse(
            success=True,
            titles=titles,
            message=f"Successfully generated {len(titles)} SEO titles",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent-info")
async def agent_info():
    """Return agent information for Telex.im discovery."""
    return {
        "name": "SEO Title Generator",
        "description": "AI agent that generates 10 SEO-optimized titles with descriptions for any topic",
        "version": "1.0.0",
        "ai_model": "Google Gemini 2.5 Flash (FREE)",
        "api_cost": "FREE - No charges",
        "capabilities": [
            "Generate SEO-optimized titles",
            "Create meta descriptions",
            "Extract relevant keywords",
            "Optimize for search engines"
        ],
        "usage": {
            "commands": [
                "generate seo titles for: [topic]",
                "seo: [topic] | keywords: [keywords]"
            ],
            "examples": [
                "generate seo titles for: sustainable fashion trends",
                "seo: AI in healthcare | keywords: machine learning, diagnosis"
            ]
        },
        "webhook_url": "/webhook",
        "protocol": "A2A"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)