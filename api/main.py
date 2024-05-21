import os
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from models import ChatModelSchema
from lang import get_response

description = """
Multi Chat with LLMs
"""

app = FastAPI(title="LLM chat API", version="0.0.1", description=description)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"Hello": "Word"}

@app.post("/stream/")
async def stream(request: Request):
    req = await request.json()
    user_query = req["user_query"]
    chat_history = req["chat_history"]
    return StreamingResponse(get_response(user_query, chat_history))