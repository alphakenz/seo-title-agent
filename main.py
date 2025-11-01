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

def generate_seo_titles(topic: str, keywords: Optional[str] = None) -> List[SEOTitle]:
    """Generate 10 SEO-optimized titles with descriptions using Google Gemini (FREE)."""
    
    if not model:
        raise HTTPException(status_code=500, detail="AI model not initialized. Please set GEMINI_API_KEY")
    
    prompt = f"""Generate 10 SEO-optimized titles with descriptions for the following topic:

Topic: {topic}
{f'Keywords to include: {keywords}' if keywords else ''}

Requirements:
1. Each title should be 50-60 characters (optimal for search engines)
2. Include power words and numbers where appropriate
3. Make titles click-worthy but accurate
4. Each description should be 150-160 characters (meta description length)
5. Descriptions should complement the title and include a call-to-action
6. Extract 3-5 relevant keywords for each title

Return the response in JSON format as an array of objects with this structure:
{{
  "title": "SEO-optimized title here",
  "description": "Engaging meta description here",
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}

Make sure all titles are unique, engaging, and optimized for search engines. Return ONLY valid JSON array, no additional text."""

    try:
        logger.info(f"Generating titles for topic: {topic}")
        response = model.generate_content(prompt)
        content = response.text
        logger.info("✅ Received response from Gemini API")
        
        # Extract JSON from response
        start_idx = content.find('[')
        end_idx = content.rfind(']') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON array found in response")
        
        json_content = content[start_idx:end_idx]
        titles_data = json.loads(json_content)
        
        # Convert to SEOTitle objects
        seo_titles = []
        for item in titles_data[:10]:  # Ensure only 10 titles
            seo_titles.append(SEOTitle(
                title=item['title'],
                description=item['description'],
                character_count=len(item['title']),
                keywords=item.get('keywords', [])
            ))
        
        logger.info(f"✅ Successfully generated {len(seo_titles)} titles")
        return seo_titles
        
    except Exception as e:
        logger.error(f"Error generating titles: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate titles: {str(e)}")

def format_titles_for_telex(titles: List[SEOTitle]) -> str:
    """Format SEO titles for display in Telex chat."""
    
    message = "🎯 **SEO Title Generator Results**\n\n"
    message += f"Generated {len(titles)} optimized titles:\n\n"
    
    for i, title in enumerate(titles, 1):
        message += f"**{i}. {title.title}**\n"
        message += f"📝 {title.description}\n"
        message += f"🔤 Length: {title.character_count} chars\n"
        message += f"🏷️ Keywords: {', '.join(title.keywords)}\n\n"
    
    message += "\n💡 **Tips:**\n"
    message += "- Titles are optimized for 50-60 characters\n"
    message += "- Descriptions fit meta description length (150-160 chars)\n"
    message += "- Use A/B testing to find the best performer\n"
    
    return message

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "agent": "SEO Title Generator",
        "version": "1.0.0",
        "description": "AI agent that generates 10 SEO-optimized titles with descriptions",
        "ai_model": "Google Gemini 1.5 Flash (FREE)",
        "api_configured": model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ai_model": "Google Gemini 1.5 Flash (FREE)",
        "ai_client": "connected" if model else "disconnected"
    }

@app.post("/webhook")
async def telex_webhook(request: TelexRequest):
    """Main webhook endpoint for Telex.im A2A protocol."""
    
    try:
        message_content = request.message.content.strip()
        channel_id = request.message.channel_id
        
        logger.info(f"Received message from channel {channel_id}: {message_content}")
        
        # Parse the message to extract topic and keywords
        if not any(trigger in message_content.lower() for trigger in ['seo', 'title', 'generate']):
            return {
                "response": "👋 Hi! I'm the SEO Title Generator Agent (powered by FREE Google Gemini 1.5).\n\n"
                           "I can generate 10 SEO-optimized titles with descriptions for any topic.\n\n"
                           "**Usage:**\n"
                           "- `generate seo titles for: your topic here`\n"
                           "- `seo: your topic | keywords: keyword1, keyword2`\n\n"
                           "Example: `generate seo titles for: sustainable fashion trends 2025`"
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
                           "Example: `generate seo titles for: AI in healthcare`"
            }
        
        # Generate SEO titles
        titles = generate_seo_titles(topic, keywords)
        
        # Format response for Telex
        formatted_message = format_titles_for_telex(titles)
        
        return {
            "response": formatted_message,
            "metadata": {
                "titles_generated": len(titles),
                "topic": topic,
                "ai_model": "Google Gemini 1.5 Flash",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return {
            "response": f"❌ Error generating titles: {str(e)}\n\n"
                       "Please try again or contact support if the issue persists."
        }

@app.post("/generate")
async def generate_titles_api(topic: str, keywords: Optional[str] = None):
    """Direct API endpoint for generating SEO titles (for testing)."""
    
    try:
        titles = generate_seo_titles(topic, keywords)
        
        return AgentResponse(
            success=True,
            titles=titles,
            message=f"Successfully generated {len(titles)} SEO titles using Google Gemini 1.5 Flash",
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
        "ai_model": "Google Gemini 1.5 Flash",
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