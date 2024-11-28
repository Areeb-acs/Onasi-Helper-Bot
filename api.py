from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from backend.core import run_llm
from langchain_groq import ChatGroq
import json
import os
from dotenv import load_dotenv
import requests
import base64
import logging
from uuid import uuid4  # For generating unique session IDs

import requests



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



def update_s3_file(new_content):
    """Upload or update the file in S3."""
    session_file_url = "https://onasi-chatbot.s3.us-east-1.amazonaws.com/conversations.txt"
    try:
        # Fetch current content
        response = requests.get(session_file_url)
        current_content = response.text if response.status_code == 200 else ""

        # Append new content
        updated_content = current_content + new_content

        # Upload updated content
        response = requests.put(session_file_url, data=updated_content)
        if response.status_code == 200:
            logging.info(f"S3 file updated successfully!")
        else:
            logging.error(f"Error updating file in S3: {response.status_code}")
    except Exception as e:
        logging.error(f"Error updating file in S3: {str(e)}")


# Initialize ChatGroq
groq_api_key = os.getenv("GROQ_API_KEY")
chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")


# HTML formatting prompt template
HTML_PROMPT_TEMPLATE = """
Please use the answer and keep it exactly the same, just change to html format ONLY.
Answer to format: {answer}
"""


# Load FAQ data
with open("./faq_data.json", "r") as f:
    faq_data = json.load(f)




def log_conversation(user_query, ai_response):
    """Logs the conversation to the public S3 file."""
    new_entry = f"User: {user_query}\nAI: {ai_response}\n{'=' * 50}\n"
    update_s3_file(new_entry)
    
def format_chat_history(chat_history):
    """
    Format the chat history into a readable string for inclusion in LLM input.
    Supports lists of dictionaries or tuples.
    """
    if not chat_history:
        return "No previous history."

    formatted_history = ""
    for i, entry in enumerate(chat_history):
        if isinstance(entry, dict):  # If the entry is a dictionary
            user_input = entry.get("user", "No input")
            ai_response = entry.get("ai", "No response")
        elif isinstance(entry, tuple) and len(entry) == 2:  # If the entry is a tuple with two elements
            user_input, ai_response = entry
        else:  # Fallback for unexpected structures
            user_input = "Unknown format"
            ai_response = "Unknown format"

        formatted_history += f"\nUser: {user_input}\nAI: {ai_response}\n"
    
    return formatted_history.strip()

@app.get("/")
async def get_root():
    return {"message": "Welcome to Onasi Helper Bot!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    question = data.get("question")
    domain = data.get("domain", None)
    chat_history = data.get("chat_history", [])  # Read chat_history from the request

    # if not question:
    #     raise HTTPException(status_code=400, detail="Question is required.")

    # Check if chat_history is empty
    if not chat_history:
        session_id = str(uuid4())  # Generate a new session ID
        logging.info(f"Generated new session ID: {session_id}")
        chat_history = [{"user": f"New session initialized with ID: {session_id}", "ai": "Welcome! How can I assist you?"}]

        # Log new session in S3
        new_session_entry = f"New Session Initialized: {session_id}\n{'=' * 50}\n"
        update_s3_file(new_session_entry)

    # Determine domain if not provided
    if not domain:
        if "rcm" in question.lower():
            domain = "RCM"
        elif "dhis" in question.lower():
            domain = "DHIS"
        else:
            domain = None

    if domain and domain.upper() not in SUPPORTED_DOMAINS:
        return {"error": f"Unsupported domain '{domain}'."}

    # Format chat history into a string
    formatted_chat_history = format_chat_history(chat_history)

    # Check FAQ first
    for qa_pair in faq_data:
        if question.lower() in qa_pair["question"].lower():
            raw_answer = qa_pair["answer"]

            # Log the conversation
            log_conversation(question, raw_answer)

            # Format response for HTML
            formatted_response = chat.invoke(
                HTML_PROMPT_TEMPLATE.format(answer=raw_answer)
            )
            return {
                "response": HTMLResponse(content=formatted_response.content)
            }

    # Proceed with LLM processing
    async def response_generator():
        # Pass the formatted chat history to the run_llm function
        generated_response = run_llm(query=question, chat_history=formatted_chat_history, domain=domain)
        answer = generated_response.get("answer", "")

        # Log the conversation
        log_conversation(question, answer)
        # Yield chunks of the response for streaming
        for chunk in answer:
            yield chunk

    logging.info(f"Received query: {question}, Domain: {domain}, Chat History: {chat_history}")
    return StreamingResponse(response_generator(), media_type="text/plain")