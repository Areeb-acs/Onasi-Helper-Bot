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









def get_session_data(bucket_name, file_url, session_id):
    """
    Fetches the conversation data for a specific session ID from a publicly accessible S3 file.

    Parameters:
    - bucket_name (str): The name of the S3 bucket (not directly needed for the request).
    - file_url (str): The full URL to the file in the S3 bucket.
    - session_id (str): The session ID to filter the data.

    Returns:
    - str: The conversation data for the specified session ID.
    """
    try:
        response = requests.get(file_url)
        response.raise_for_status()  # Raise an error for unsuccessful requests
        
        # Split the file into sessions
        sessions = response.text.split("==================================================")
        
        # Filter sessions for the given session_id
        filtered_data = [
            session.strip()
            for session in sessions
            if f"Session ID: {session_id}" in session
        ]
        
        # Join the filtered sessions back into a single string
        return "\n==================================================\n".join(filtered_data)
    
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"










def fetch_s3_file(session_id):
    """Fetch interactions matching the given session ID from the public S3 file."""
    chat_data = get_session_data(BUCKET_NAME, FILE_URL, session_id)
    return chat_data


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

        formatted_history += f"\nTurn {i+1}:\nUser: {user_input}\nAI: {ai_response}\n"
    
    return formatted_history.strip()


def update_s3_file(session_id, new_content):
    """Upload or update session-specific file in S3."""
    session_file_url = "https://onasi-chatbot.s3.us-east-1.amazonaws.com/conversations.txt"
    try:
        # Fetch current session-specific content
        response = requests.get(session_file_url)
        current_content = response.text if response.status_code == 200 else ""

        # Append new content
        updated_content = current_content + new_content

        # Upload updated content
        response = requests.put(session_file_url, data=updated_content)
        if response.status_code == 200:
            logging.info(f"S3 file updated successfully for session {session_id}!")
        else:
            logging.error(f"Error updating session file in S3: {response.status_code}")
    except Exception as e:
        logging.error(f"Error updating session file in S3: {str(e)}")

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



def get_last_ten_conversations(session_id):
    """Reads the last ten interactions for the specified session ID."""
    file_content = fetch_s3_file(session_id)
    logging.debug(f"S3 file content:\n{file_content}")

    if not file_content:
        return []

    # Split interactions using the consistent separator
    interactions = file_content.strip().split("=" * 50)
    logging.debug(f"Split interactions:\n{interactions}")

    # Filter relevant interactions for the specific session ID
    relevant_interactions = [
        interaction.strip()
        for interaction in interactions
        if f"Session ID: {session_id}" in interaction
    ]
    logging.debug(f"Filtered interactions for session {session_id}:\n{relevant_interactions}")

    # Take the last 10 relevant interactions
    last_ten = relevant_interactions[-10:]

    # Parse user and AI responses into a structured chat history
    chat_history = []
    for interaction in last_ten:
        user_line = next((line for line in interaction.splitlines() if line.startswith("User:")), None)
        ai_line = next((line for line in interaction.splitlines() if line.startswith("AI:")), None)
        if user_line and ai_line:
            chat_history.append({
                "user": user_line.replace("User: ", "").strip(),
                "ai": ai_line.replace("AI: ", "").strip()
            })

    logging.debug(f"Parsed chat history for session {session_id}:\n{chat_history}")
    return chat_history

def log_conversation(session_id, user_query, ai_response):
    """Logs the conversation to the public S3 file."""
    new_entry = f"Session ID: {session_id}\nUser: {user_query}\nAI: {ai_response}\n{'=' * 50}\n"
    update_s3_file(session_id, new_entry)


@app.get("/")
async def get_root():
    return {"message": "Welcome to Onasi Helper Bot!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    question = data.get("question")
    session_id = data.get("session_id")
    domain = data.get("domain", None)

    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    if not session_id:
        session_id = str(uuid4())
        logging.info(f"Generated new session_id: {session_id}")

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

    # Fetch chat history
    chat_history_content = fetch_s3_file(session_id)
    formatted_history = format_chat_history(chat_history_content)
    print(formatted_history)
    # Check FAQ first
    for qa_pair in faq_data:
        if question.lower() in qa_pair["question"].lower():
            raw_answer = qa_pair["answer"]
            log_conversation(session_id, question, raw_answer)

            # Format response for HTML
            formatted_response = chat.invoke(
                HTML_PROMPT_TEMPLATE.format(answer=raw_answer)
            )
            return {
                "session_id": session_id,  # Always return session_id
                "response": HTMLResponse(content=formatted_response.content)
            }

    # Proceed with LLM processing
    async def response_generator():
        generated_response = run_llm(query=question, chat_history=formatted_history, domain=domain)
        answer = generated_response.get("answer", "")

        log_conversation(session_id, question, answer)
        update_s3_file(session_id, answer)

        # Yield chunks of the response for streaming
        for chunk in answer:
            yield chunk

    logging.info(f"Session ID: {session_id}, Received query: {question}, Domain: {domain}")
    return StreamingResponse(response_generator(), media_type="text/plain")


