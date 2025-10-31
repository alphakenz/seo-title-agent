"""
AI Code Review Agent for Telex.im
A2A Protocol Implementation using FastAPI
"""

from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import uuid
import asyncio
import json
import os
from openai import AsyncOpenAI

# Initialize FastAPI app
app = FastAPI(
    title="AI Code Review Agent",
    description="Intelligent code review agent using A2A protocol",
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

# OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# In-memory storage
tasks_db: Dict[str, Dict] = {}
contexts_db: Dict[str, List[Dict]] = {}

# Security
VALID_TOKENS = os.getenv("AGENT_TOKENS", "demo-token-12345").split(",")


# Pydantic Models
class MessagePart(BaseModel):
    kind: Literal["text", "file", "data"]
    text: Optional[str] = None
    mimeType: Optional[str] = None
    data: Optional[str] = None
    name: Optional[str] = None


class Message(BaseModel):
    role: Literal["user", "agent"]
    parts: List[MessagePart]
    messageId: Optional[str] = None
    timestamp: Optional[str] = None


class TaskState(BaseModel):
    status: Literal["submitted", "working", "completed", "failed", "canceled"]
    message: Optional[str] = None
    progress: Optional[float] = None


class Task(BaseModel):
    taskId: str
    contextId: Optional[str] = None
    state: TaskState
    input: Message
    output: Optional[Message] = None
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    createdAt: str
    updatedAt: str


class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any]
    id: Optional[str] = None


class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


# Authentication dependency
async def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        if token not in VALID_TOKENS:
            raise HTTPException(status_code=401, detail="Invalid token")
        return token
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")


# Agent Card endpoint
@app.get("/.well-known/agent-card.json")
async def agent_card():
    return {
        "name": "AI Code Review Agent",
        "description": "An intelligent code review assistant that analyzes code, suggests improvements, identifies bugs, and provides security recommendations using GPT-4",
        "url": os.getenv("AGENT_URL", "https://your-agent-url.com"),
        "version": "1.0.0",
        "capabilities": {
            "streaming": True,
            "multiTurn": True,
            "artifacts": True
        },
        "security": [
            {
                "type": "http",
                "scheme": "bearer",
                "description": "Bearer token authentication"
            }
        ],
        "skills": [
            {
                "name": "code_review",
                "description": "Comprehensive code review with best practices",
                "examples": [
                    "Review this Python function for me",
                    "Check this JavaScript code for security issues",
                    "Suggest improvements for this SQL query"
                ]
            },
            {
                "name": "bug_detection",
                "description": "Identify potential bugs and errors",
                "examples": [
                    "Find bugs in this code",
                    "What's wrong with this implementation?"
                ]
            },
            {
                "name": "optimization",
                "description": "Suggest performance optimizations",
                "examples": [
                    "How can I make this code faster?",
                    "Optimize this algorithm"
                ]
            }
        ],
        "endpoint": {
            "url": "/a2a/v1",
            "protocol": "json-rpc-2.0"
        }
    }


# Core A2A endpoint
@app.post("/a2a/v1")
async def a2a_endpoint(
    request: JsonRpcRequest,
    token: str = Depends(verify_token)
):
    try:
        if request.method == "message/send":
            return await handle_message_send(request.params, request.id)
        elif request.method == "message/stream":
            return await handle_message_stream(request.params, request.id)
        elif request.method == "tasks/get":
            return await handle_task_get(request.params, request.id)
        elif request.method == "tasks/cancel":
            return await handle_task_cancel(request.params, request.id)
        else:
            return JsonRpcResponse(
                error={
                    "code": -32601,
                    "message": f"Method not found: {request.method}"
                },
                id=request.id
            )
    except Exception as e:
        return JsonRpcResponse(
            error={
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            },
            id=request.id
        )


async def handle_message_send(params: Dict, request_id: Optional[str]):
    """Handle message/send requests"""
    message = Message(**params.get("message", {}))
    context_id = params.get("contextId", str(uuid.uuid4()))
    
    # Create task
    task_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"
    
    task = Task(
        taskId=task_id,
        contextId=context_id,
        state=TaskState(status="submitted"),
        input=message,
        createdAt=now,
        updatedAt=now
    )
    
    tasks_db[task_id] = task.dict()
    
    # Store context
    if context_id not in contexts_db:
        contexts_db[context_id] = []
    contexts_db[context_id].append(message.dict())
    
    # Process asynchronously
    asyncio.create_task(process_code_review(task_id, message, context_id))
    
    return JsonRpcResponse(
        result={"task": tasks_db[task_id]},
        id=request_id
    )


async def process_code_review(task_id: str, message: Message, context_id: str):
    """Process code review using OpenAI"""
    try:
        # Update to working
        tasks_db[task_id]["state"] = {"status": "working", "progress": 0.1}
        tasks_db[task_id]["updatedAt"] = datetime.utcnow().isoformat() + "Z"
        
        # Extract code from message
        user_text = ""
        for part in message.parts:
            if part.kind == "text" and part.text:
                user_text += part.text + "\n"
        
        # Get conversation history
        history = contexts_db.get(context_id, [])
        messages = [
            {
                "role": "system",
                "content": """You are an expert code reviewer. Analyze code for:
1. **Bugs & Errors**: Identify syntax errors, logic bugs, runtime issues
2. **Best Practices**: Check adherence to language conventions and patterns
3. **Security**: Flag potential security vulnerabilities
4. **Performance**: Suggest optimizations
5. **Readability**: Recommend improvements for clarity

Format your response with clear sections using markdown."""
            }
        ]
        
        # Add history (last 5 messages)
        for msg in history[-5:]:
            role = "user" if msg["role"] == "user" else "assistant"
            text = " ".join([p.get("text", "") for p in msg.get("parts", [])])
            messages.append({"role": role, "content": text})
        
        # Call OpenAI
        tasks_db[task_id]["state"]["progress"] = 0.5
        
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        review_text = response.choices[0].message.content
        
        # Create output message
        output_message = Message(
            role="agent",
            parts=[
                MessagePart(
                    kind="text",
                    text=review_text
                )
            ],
            messageId=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        # Create artifact if code was reviewed
        artifacts = []
        if "```" in user_text:
            artifacts.append({
                "name": "code_review_report.md",
                "mimeType": "text/markdown",
                "data": review_text,
                "description": "Comprehensive code review report"
            })
        
        # Update task to completed
        tasks_db[task_id].update({
            "state": {"status": "completed", "progress": 1.0},
            "output": output_message.dict(),
            "artifacts": artifacts,
            "updatedAt": datetime.utcnow().isoformat() + "Z"
        })
        
        # Add to context
        contexts_db[context_id].append(output_message.dict())
        
    except Exception as e:
        tasks_db[task_id].update({
            "state": {
                "status": "failed",
                "message": f"Error: {str(e)}"
            },
            "updatedAt": datetime.utcnow().isoformat() + "Z"
        })


async def handle_message_stream(params: Dict, request_id: Optional[str]):
    """Handle streaming responses"""
    message = Message(**params.get("message", {}))
    context_id = params.get("contextId", str(uuid.uuid4()))
    
    async def generate():
        # Extract text
        user_text = " ".join([p.text or "" for p in message.parts if p.kind == "text"])
        
        # Stream from OpenAI
        stream = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a code review expert."},
                {"role": "user", "content": user_text}
            ],
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                data = {
                    "type": "content",
                    "data": chunk.choices[0].delta.content
                }
                yield f"data: {json.dumps(data)}\n\n"
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


async def handle_task_get(params: Dict, request_id: Optional[str]):
    """Retrieve task status"""
    task_id = params.get("taskId")
    
    if task_id not in tasks_db:
        return JsonRpcResponse(
            error={
                "code": -32602,
                "message": f"Task not found: {task_id}"
            },
            id=request_id
        )
    
    return JsonRpcResponse(
        result={"task": tasks_db[task_id]},
        id=request_id
    )


async def handle_task_cancel(params: Dict, request_id: Optional[str]):
    """Cancel a task"""
    task_id = params.get("taskId")
    
    if task_id not in tasks_db:
        return JsonRpcResponse(
            error={
                "code": -32602,
                "message": f"Task not found: {task_id}"
            },
            id=request_id
        )
    
    tasks_db[task_id]["state"] = {"status": "canceled"}
    tasks_db[task_id]["updatedAt"] = datetime.utcnow().isoformat() + "Z"
    
    return JsonRpcResponse(
        result={"task": tasks_db[task_id]},
        id=request_id
    )


# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)