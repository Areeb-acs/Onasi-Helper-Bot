from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from langchain_groq import ChatGroq
import json
import os
from dotenv import load_dotenv
import requests
import base64
import logging
from uuid import uuid4  # For generating unique session IDs
from langchain_pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from fastapi.responses import JSONResponse
from fastapi import HTTPException

import requests

from claimscrubber import analyze_fhir_message_and_get_explanation, chat_helper

# Load environment variables
load_dotenv()

# GitHub Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")
FILE_PATH = os.getenv("FILE_PATH")
BRANCH = os.getenv("BRANCH", "main")

app = FastAPI()
# File to store conversations
CONVERSATION_LOG_FILE = "conversations.txt"
SUPPORTED_DOMAINS = {"RCM", "DHIS"}

# A global dictionary to store session-specific chat histories in memory
session_chat_histories = {}


BUCKET_NAME = "onasi-chatbot"
FILE_URL = "https://onasi-chatbot.s3.us-east-1.amazonaws.com/conversations.txt"
INDEX_NAME = "rcm-final-app"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

groq_api_key = os.getenv("GROQ_API_KEY")
# Initialize Pinecone

# Initialize ChatGroq
groq_api_key = os.getenv("GROQ_API_KEY")
chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

# FastAPI root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the FHIR Claim Scrubber API!"}

# FastAPI POST endpoint to analyze FHIR messages
# FastAPI POST endpoint to analyze FHIR messages
@app.post("/analyze-fhir/")
async def analyze_fhir_endpoint(request: Request):
    """
    API endpoint to analyze FHIR messages and explain the results.

    Expects the FHIR message as plain text in the request body.

    Returns:
    {
        "analysis_result": "<analysis result>",
        "explanation": "<detailed explanation>"
    }
    """
    try:
        # Read the FHIR content from the request body
        fhir_content = await request.body()
        fhir_content = fhir_content.decode("utf-8").strip()

        if not fhir_content:
            raise HTTPException(status_code=400, detail="FHIR content is required in the request body.")

        logging.info("FHIR message content loaded successfully.")

        # Analyze the FHIR message
        analysis_result = analyze_fhir_message_and_get_explanation(fhir_content)

        # Generate explanation using chat model
        explanation = chat_helper(chat, analysis_result)

        return JSONResponse(content={"analysis_result": analysis_result, "explanation": explanation})

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")