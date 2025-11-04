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

# Load environment variables from .env file
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

if not GEMINI_API_KEY and not DISABLE_AI:
    logger.warning("⚠️ GEMINI_API_KEY not set. Agent will use fallback mode.")
    logger.warning("Get FREE API key at: https://makersuite.google.com/app/apikey")
    model = None
else:
    if DISABLE_AI:
        logger.info("AI generation disabled via DISABLE_AI flag")
        model = None
    else:
        logger.info("✅ GEMINI_API_KEY loaded successfully")
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Try to initialize Gemini model
        preferred_models = [
            'gemini-2.0-flash-exp',
            'gemini-1.5-flash',
            'gemini-1.5-flash-latest',
            'gemini-pro'
        ]
        
        model = None
        working_model_name = None
        
        for model_name in preferred_models:
            try:
                logger.info(f"🔍 Trying model: {model_name}")
                test_model = genai.GenerativeModel(model_name)
                test_response = test_model.generate_content(
                    "Say OK",
                    generation_config=genai.GenerationConfig(max_output_tokens=10)
                )
                logger.info(f"✅ Model {model_name} working!")
                model = test_model
                working_model_name = model_name
                break
            except Exception as e:
                logger.warning(f"⚠️ {model_name} failed: {str(e)[:100]}")
                continue
        
        if model:
            logger.info(f"🎉 Successfully initialized: {working_model_name}")
        else:
            logger.error("❌ All models failed. Using fallback mode.")

def generate_fallback_titles(topic: str, keywords: Optional[str] = None) -> List[SEOTitle]:
    """Generate simple fallback titles if AI is unavailable."""
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
    for title in base_titles:
        seo_titles.append(SEOTitle(
            title=title[:60],
            description=f"Discover proven strategies and expert insights for {topic}. Get actionable tips and best practices to achieve your goals!",
            character_count=len(title[:60]),
            keywords=keyword_list[:3]
        ))
    
    logger.info(f"✅ Generated {len(seo_titles)} fallback titles")
    return seo_titles

def generate_seo_titles(topic: str, keywords: Optional[str] = None) -> List[SEOTitle]:
    """Generate 10 SEO-optimized titles using Google Gemini."""
    
    if DISABLE_AI or not model:
        logger.info("Using fallback titles (AI disabled or not initialized)")
        return generate_fallback_titles(topic, keywords)
    
    keyword_hint = f"\nKeywords to include: {keywords}" if keywords else ""
    
    prompt = f"""Generate 10 SEO-optimized titles for the topic: "{topic}"{keyword_hint}

Return ONLY valid JSON array with this exact format (no markdown, no extra text):
[{{"title":"Example Title Here","description":"Example description approximately 150 characters long.","keywords":["keyword1","keyword2","keyword3"]}}]

Requirements:
- Return exactly 10 title objects
- Each title: 50-60 characters maximum
- Each description: 140-160 characters
- Each item has exactly 3 relevant keywords
- Pure JSON array only - no markdown code blocks, no explanatory text
- Valid JSON syntax with proper escaping"""

    try:
        logger.info(f"Generating titles for topic: {topic}")
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=2000,  # Reduced for faster response
                temperature=0.7,
                top_p=0.8,  # Reduced for faster response
                top_k=20    # Reduced for faster response
            )
        )
        
        content = response.text.strip()
        logger.info("✅ Received response from Gemini API")
        
        # Clean the response
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        content = content.replace('`', '')
        
        # Find JSON array
        start_idx = content.find('[')
        end_idx = content.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            logger.error("No valid JSON array found in response")
            return generate_fallback_titles(topic, keywords)
        
        json_content = content[start_idx:end_idx + 1]
        
        # Parse JSON
        try:
            titles_data = json.loads(json_content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            # Try to fix common JSON issues
            json_content = json_content.replace("'", '"')
            json_content = re.sub(r',\s*}', '}', json_content)
            json_content = re.sub(r',\s*]', ']', json_content)
            
            try:
                titles_data = json.loads(json_content)
                logger.info("✅ Fixed JSON successfully")
            except json.JSONDecodeError:
                logger.error("Could not fix JSON, using fallback")
                return generate_fallback_titles(topic, keywords)
        
        if not isinstance(titles_data, list):
            logger.error(f"Response is not a list: {type(titles_data)}")
            return generate_fallback_titles(topic, keywords)
        
        # Convert to SEOTitle objects
        seo_titles = []
        for item in titles_data[:10]:
            try:
                title_text = item.get('title', 'SEO Title')[:60]
                desc_text = item.get('description', 'Description here.')[:160]
                kw_list = item.get('keywords', [topic, 'guide', 'tips'])[:3]
                
                seo_titles.append(SEOTitle(
                    title=title_text,
                    description=desc_text,
                    character_count=len(title_text),
                    keywords=kw_list
                ))
            except Exception as e:
                logger.warning(f"Skipping invalid title: {str(e)}")
                continue
        
        # Ensure we have enough titles
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
    
    message = "✨ **Your 10 SEO-Optimized Titles:**\n\n"
    
    for i, title in enumerate(titles, 1):
        message += f"**{i}. {title.title}**\n"
        message += f"📝 {title.description}\n"
        message += f"🏷️ Keywords: {', '.join(title.keywords[:3])} • {title.character_count} chars\n\n"
    
    message += "💡 **Pro Tips:**\n"
    message += "• Test multiple titles for best CTR\n"
    message += "• Use power words and numbers\n"
    message += "• Keep titles under 60 characters\n"
    message += "• A/B test your favorites\n"
    
    return message

@app.get("/")
async def root():
    """Root endpoint with agent information."""
    return {
        "status": "online",
        "agent": "SEO Title Generator",
        "version": "1.0.0",
        "description": "AI-powered SEO title generator that creates 10 optimized titles with descriptions",
        "ai_model": "Google Gemini (FREE)",
        "ai_status": "connected" if model else "fallback mode",
        "endpoints": {
            "health": "/health",
            "webhook": "/webhook",
            "generate": "/generate",
            "info": "/agent-info"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ai_model": "Google Gemini",
        "ai_status": "connected" if model else "fallback",
        "version": "1.0.0"
    }

@app.get("/webhook")
async def webhook_health():
    """Webhook health check - Telex checks this first."""
    return {
        "status": "ready",
        "message": "Webhook endpoint is active",
        "endpoint": "/webhook",
        "method": "POST",
        "agent": "SEO Title Generator",
        "version": "1.0.0"
    }

@app.post("/webhook")
async def telex_webhook(request: Request):
    """Main webhook endpoint for Telex.im integration."""
    
    try:
        body = await request.json()
        logger.info(f"📨 Received webhook request")
        logger.info(f"📦 Request body: {json.dumps(body, indent=2)}")
        
        # Extract message content - Handle Telex A2A format
        message_content = ""
        
        # Try different Telex message formats
        if "params" in body and "message" in body["params"]:
            # A2A JSON-RPC format
            message_obj = body["params"]["message"]
            if "parts" in message_obj:
                # Extract text from parts array
                for part in message_obj["parts"]:
                    if part.get("kind") == "text":
                        message_content = part.get("text", "")
                        break
            elif "content" in message_obj:
                message_content = message_obj.get("content", "")
        elif "message" in body:
            # Direct message format
            if isinstance(body["message"], dict):
                if "parts" in body["message"]:
                    # A2A format with parts
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
        elif "content" in body:
            message_content = body.get("content", "")
        elif "text" in body:
            message_content = body.get("text", "")
        else:
            logger.error(f"❌ Unknown request format: {body.keys()}")
            return {
                "response": "❌ Unable to parse message. Please try again."
            }
        
        message_content = message_content.strip()
        logger.info(f"📝 Processing: {message_content[:100]}")
        
        # Check if it's a help/info request
        if not any(trigger in message_content.lower() for trigger in ['seo', 'title', 'generate', 'create']):
            return {
                "response": "👋 **Welcome to SEO Title Generator!**\n\n"
                           "🎯 I create 10 SEO-optimized titles that rank!\n\n"
                           "**How to use:**\n"
                           "• `generate seo titles for: [your topic]`\n"
                           "• `seo: [topic] | keywords: [word1, word2]`\n\n"
                           "**Examples:**\n"
                           "• `generate seo titles for: AI in healthcare`\n"
                           "• `seo: Python programming | keywords: tutorial, beginner, code`\n\n"
                           "Ready to boost your SEO? Let's go! 🚀"
            }
        
        # Extract topic and keywords
        topic = message_content
        keywords = None
        
        # Handle format: "topic | keywords: word1, word2"
        if '|' in message_content:
            parts = message_content.split('|')
            topic = parts[0].strip()
            if len(parts) > 1 and 'keywords:' in parts[1].lower():
                keywords = parts[1].split('keywords:', 1)[1].strip()
        
        # Clean up topic by removing common prefixes
        prefixes_to_remove = [
            'generate seo titles for:',
            'seo titles for:',
            'create seo titles for:',
            'generate titles for:',
            'create titles for:',
            'seo:',
            'generate:',
            'create:'
        ]
        
        topic_lower = topic.lower()
        for prefix in prefixes_to_remove:
            if topic_lower.startswith(prefix):
                topic = topic[len(prefix):].strip()
                break
        
        # Validate topic
        if not topic or len(topic) < 3:
            return {
                "response": "❌ **Please provide a valid topic!**\n\n"
                           "Example: `generate seo titles for: sustainable fashion`"
            }
        
        logger.info(f"🎯 Generating titles for: {topic}")
        if keywords:
            logger.info(f"🏷️ With keywords: {keywords}")
        
        # Generate titles with timeout (reduced to 20s for Telex compatibility)
        try:
            titles = await asyncio.wait_for(
                asyncio.to_thread(generate_seo_titles, topic, keywords),
                timeout=20.0
            )
        except asyncio.TimeoutError:
            logger.error("⏱️ Generation timed out")
            return {
                "response": "⏱️ **Request timed out**\n\n"
                           "Please try a simpler topic or try again."
            }
        except Exception as gen_error:
            logger.error(f"❌ Generation error: {str(gen_error)}", exc_info=True)
            return {
                "response": "❌ **Error generating titles**\n\n"
                           "Please try again with a different topic."
            }
        
        # Format and return response
        formatted_message = format_titles_for_telex(titles)
        
        return {
            "response": formatted_message,
            "metadata": {
                "titles_generated": len(titles),
                "topic": topic,
                "keywords": keywords,
                "ai_model": "Google Gemini",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing webhook: {str(e)}", exc_info=True)
        return {
            "response": "❌ An error occurred while processing your request. Please try again."
        }

@app.post("/generate")
async def generate_titles_api(request: GenerateRequest):
    """Direct API endpoint for generating SEO titles."""
    
    try:
        titles = generate_seo_titles(request.topic, request.keywords)
        
        return AgentResponse(
            success=True,
            titles=titles,
            message=f"Successfully generated {len(titles)} SEO titles",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent-info")
async def agent_info():
    """Agent information for Telex.im discovery."""
    return {
        "name": "SEO Title Generator",
        "description": "AI agent that generates 10 SEO-optimized titles with descriptions for any topic",
        "version": "1.0.0",
        "ai_model": "Google Gemini (FREE)",
        "api_cost": "FREE - No charges for Gemini API",
        "capabilities": [
            "Generate SEO-optimized titles (50-60 chars)",
            "Create meta descriptions (150-160 chars)",
            "Extract relevant keywords",
            "Optimize for search engines and CTR"
        ],
        "usage": {
            "commands": [
                "generate seo titles for: [topic]",
                "seo: [topic] | keywords: [keywords]"
            ],
            "examples": [
                "generate seo titles for: sustainable fashion trends",
                "seo: AI in healthcare | keywords: machine learning, diagnosis",
                "create seo titles for: WordPress SEO plugins"
            ]
        },
        "endpoints": {
            "webhook": "/webhook",
            "health": "/health",
            "generate": "/generate (POST with JSON)",
            "info": "/agent-info"
        },
        "protocol": "Telex Webhook",
        "author": "Built for HNG Internship",
        "repository": "https://github.com/alphakenz/seo-title-agent"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting SEO Title Generator on {host}:{port}")
    uvicorn.run(app, host=host, port=port)