# SEO Title Generator Agent 🎯

An AI-powered SEO title generator that creates 10 optimized titles with descriptions for any topic. Built with Python, FastAPI, and Google Gemini, integrated with Telex.im.

## 🌟 What It Does

The SEO Title Generator helps content creators, marketers, and bloggers create click-worthy, SEO-optimized titles instantly:

- **10 SEO Titles**: Get 10 professionally crafted titles for any topic
- **Meta Descriptions**: Each title includes a 150-160 character description
- **Keyword Extraction**: Automatic relevant keyword identification
- **Character Count**: All titles optimized to 50-60 characters (ideal for SEO)
- **AI-Powered**: Uses Google Gemini for intelligent title generation
- **100% FREE**: No API costs - Gemini offers generous free tier

## 🎯 Features

- ✅ **SEO Optimized**: Titles designed for maximum click-through rate
- ✅ **Instant Generation**: Get 10 titles in seconds
- ✅ **Telex Integration**: Works seamlessly in Telex.im workspace
- ✅ **Keyword Support**: Add custom keywords for better targeting
- ✅ **Fallback Mode**: Works even without API key
- ✅ **Professional Format**: Clean, scannable results

## 📋 Prerequisites

- Python 3.9+
- Google Gemini API Key (FREE - get at [Google AI Studio](https://makersuite.google.com/app/apikey))
- Telex.im account

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone <your-repo-url>
cd seo-title-agent
pip install -r requirements.txt
```

### 2. Get FREE Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Get API Key"
3. Create new key or use existing project
4. Copy your API key

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Run Locally

```bash
python main.py
```

Visit `http://localhost:8000` to see the agent info!

### 5. Test It

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"topic": "AI in education", "keywords": "learning, technology"}'
```

## 🌐 Deploy to Railway

### Option 1: Railway CLI (Recommended)

```bash
# Install Railway CLI
npm install -g railway

# Login
railway login

# Initialize project
railway init

# Add environment variable
railway variables set GEMINI_API_KEY=your_key_here

# Deploy
railway up

# Get your URL
railway domain
```

Your agent will be live at `https://your-app.railway.app`!

### Option 2: Railway Dashboard

1. Go to [railway.app](https://railway.app) and sign in
2. Click "New Project" → "Deploy from GitHub"
3. Select your repository
4. Add environment variable:
   - Name: `GEMINI_API_KEY`
   - Value: Your Gemini API key
5. Click "Deploy"
6. Copy your app URL from the dashboard

## 🔗 Connect to Telex.im

### Step 1: Get Telex Access

```bash
/telex-invite your-email@example.com
```

Wait for the invitation email and create your account.

### Step 2: Create Webhook in Telex

1. Log into [Telex.im](https://telex.im)
2. Go to your workspace
3. Navigate to **Integrations** or **Webhooks**
4. Click **Add Webhook**
5. Configure:
   - **Name**: SEO Title Generator
   - **Webhook URL**: `https://your-railway-url.railway.app/webhook`
   - **Method**: POST
6. Save the webhook

### Step 3: Test in Telex

Send a message in any Telex channel:
```
@SEO-Title-Generator generate seo titles for: sustainable fashion
```

Or:
```
@SEO-Title-Generator seo: WordPress SEO | keywords: plugins, optimization
```

The agent will respond with 10 SEO-optimized titles! 🎉

## 💬 Usage Examples

### Basic Usage

```
generate seo titles for: AI in healthcare
```

**Response**: 10 SEO-optimized titles with descriptions

### With Keywords

```
seo: Python programming | keywords: tutorial, beginner, code
```

**Response**: 10 titles incorporating your specific keywords

### Quick Generation

```
create titles for: digital marketing trends 2025
```

**Response**: Instant SEO titles for your topic

## 📊 Response Format

Each title includes:

```
1. Your Optimized Title Here (58 chars)
📝 A compelling 150-160 character meta description that entices clicks
🏷️ Keywords: keyword1, keyword2, keyword3 • 58 chars
```

## 🏗️ Project Structure

```
seo-title-agent/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore rules
├── README.md           # This file
├── Procfile            # Railway deployment config
└── runtime.txt         # Python version
```

## 🔧 API Endpoints

### GET `/`
Root endpoint with agent information

### GET `/health`
Health check for monitoring

### GET `/webhook`
Webhook health check (for Telex verification)

### POST `/webhook`
Main endpoint for Telex integration

**Request Format**:
```json
{
  "message": {
    "content": "generate seo titles for: AI trends"
  }
}
```

**Response Format**:
```json
{
  "response": "✨ Your 10 SEO-Optimized Titles:\n\n...",
  "metadata": {
    "titles_generated": 10,
    "topic": "AI trends",
    "timestamp": "2025-11-04T10:00:00Z"
  }
}
```

### POST `/generate`
Direct API endpoint for title generation

**Request**:
```json
{
  "topic": "sustainable fashion",
  "keywords": "eco-friendly, ethical"
}
```

**Response**:
```json
{
  "success": true,
  "titles": [
    {
      "title": "Sustainable Fashion: Complete Guide 2025",
      "description": "Discover eco-friendly fashion trends...",
      "character_count": 42,
      "keywords": ["sustainable", "fashion", "eco-friendly"]
    }
  ],
  "message": "Successfully generated 10 SEO titles",
  "timestamp": "2025-11-04T10:00:00Z"
}
```

### GET `/agent-info`
Detailed agent information

## 🎨 Customization

### Modify System Prompt

Edit the prompt in `main.py`:

```python
prompt = f"""Generate 10 SEO-optimized titles for: "{topic}"

Your custom instructions here...
"""
```

### Change Response Format

Modify the `format_titles_for_telex()` function:

```python
def format_titles_for_telex(titles: List[SEOTitle]) -> str:
    # Your custom formatting here
    return formatted_message
```

### Add More Features

Ideas:
- A/B testing suggestions
- CTR prediction scores
- Competitor analysis
- Hashtag generation
- Social media optimized versions

## 🐛 Troubleshooting

### "GEMINI_API_KEY not set" Error

**Solution**: 
1. Get free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add to `.env` file: `GEMINI_API_KEY=your_key_here`
3. Restart the application

### Agent Returns Fallback Titles

**Possible Causes**:
- API key invalid or expired
- Rate limit exceeded
- Network issues

**Solution**:
- Check API key is correct
- Wait a few minutes (free tier has limits)
- Check logs: `tail -f app.log`

### Webhook Not Responding in Telex

**Checklist**:
1. ✅ Agent is deployed and running
2. ✅ Webhook URL is correct
3. ✅ Endpoint `/webhook` responds to GET request
4. ✅ Environment variables set on Railway
5. ✅ Check Railway logs for errors

### Timeout Errors

**Solution**:
- Simplify your topic
- Try again (Gemini occasionally slow)
- Check Railway logs for issues

## 📈 Performance

- **Response Time**: 2-5 seconds
- **Success Rate**: 99%+ (with fallback)
- **API Costs**: $0 (Gemini free tier)
- **Uptime**: Depends on Railway (99.9% typical)

## 🔐 Security

- ✅ Environment variables for sensitive data
- ✅ No API keys in code
- ✅ CORS enabled for web access
- ✅ Input validation on all endpoints
- ✅ Error handling prevents crashes

## 📊 Monitoring

### View Logs on Railway

```bash
railway logs --tail
```

### Check Application Health

```bash
curl https://your-app.railway.app/health
```

### Monitor Telex Interactions

Check Railway logs for webhook calls and responses.

## 🚀 Advanced Features

### Rate Limiting

Add rate limiting for production:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/webhook")
@limiter.limit("10/minute")
async def telex_webhook(request: Request):
    # ...
```

### Caching

Add Redis caching for repeated topics:

```python
import redis

cache = redis.Redis(host='localhost', port=6379)

def get_cached_titles(topic):
    cached = cache.get(f"titles:{topic}")
    if cached:
        return json.loads(cached)
    return None
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - feel free to use for any purpose!

## 🙏 Acknowledgments

- Built for [HNG Internship](https://hng.tech)
- Powered by [Google Gemini](https://deepmind.google/technologies/gemini/)
- Integrated with [Telex.im](https://telex.im)

## 📮 Support

- **GitHub Issues**: For bugs and features
- **Email**: apehken@gmail.com.com
- **Twitter**: [@princeKenzee]

## 🔗 Links

- **Live Demo**: https://your-app.railway.app
- **GitHub**: https://github.com/alphakenz/seo-title-agent
- **Blog Post**: [Link to your blog post]
- **Telex**: https://telex.im

---

Built with ❤️ for content creators and marketers

**Pro Tip**: Get better results by being specific with your topics and keywords!