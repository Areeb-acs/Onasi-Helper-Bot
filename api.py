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
from langchain_pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from fastapi import HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

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
INDEX_NAME = "rcm-final-app"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

groq_api_key = os.getenv("GROQ_API_KEY")
# Initialize Pinecone
docsearch = Pinecone(index_name=INDEX_NAME, embedding=embeddings)


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

# Preinitialize docsearch (can be reused across multiple queries)
docsearch = Pinecone(index_name=INDEX_NAME, embedding=embeddings)

# HTML formatting prompt template
HTML_PROMPT_TEMPLATE = """
Please use the answer and keep it exactly the same, just change to html format ONLY.
Answer to format: {answer}
"""


# Load FAQ data
with open("./faq_data.json", "r") as f:
    faq_data = json.load(f)



def get_last_10_conversations():
    """
    Fetches the last 10 Q&A pairs from the S3 conversations file.

    Returns:
        List[dict]: A list of the last 10 conversations in the format:
                    [{"user": "question1", "ai": "answer1"}, ...]
    """
    try:
        # Fetch the content of the file from the S3 bucket
        response = requests.get(FILE_URL)
        if response.status_code == 200:
            content = response.text
        else:
            logging.error(f"Failed to fetch conversation file: {response.status_code}")
            return []

        # Split content into individual conversations by the separator
        entries = content.strip().split("==================================================")
        conversations = []

        for entry in entries:
            # Process each entry to extract User and AI lines
            lines = entry.strip().split("\n")
            user_line = next((line.replace("User: ", "").strip() for line in lines if line.startswith("User:")), None)
            ai_line = next((line.replace("AI: ", "").strip() for line in lines if line.startswith("AI:")), None)

            # Append only valid entries with both User and AI content
            if user_line and ai_line:
                conversations.append({"user": user_line, "ai": ai_line})

        # Return the last 10 conversations
        return conversations[-1:] if len(conversations) > 1 else conversations

    except Exception as e:
        logging.error(f"Error fetching or parsing conversation file: {str(e)}")
        return []


def log_conversation(user_query, ai_response):
    """Logs the conversation to the public S3 file."""
    new_entry = f"User: {user_query}\nAI: {ai_response}\n{'=' * 50}\n"
    update_s3_file(new_entry)
    
def format_chat_history(chat_history):
    """
    Format the chat history into a readable string for inclusion in LLM input.
    """
    if not chat_history:
        return "No previous history."

    formatted_history = ""
    for entry in chat_history:
        user_input = entry.get("user", "Unknown input")
        ai_response = entry.get("ai", "Unknown response")
        formatted_history += f"User: {user_input}\nAI: {ai_response}\n"
    
    return formatted_history.strip()

@app.get("/")
async def get_root():
    return {"message": "Welcome to Onasi Helper Bot!"}


@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    Chat endpoint for processing user queries and returning AI-generated responses.
    """
    data = await request.json()
    question = data.get("question")
    domain = data.get("domain", None)
    chat_history = [data.get("chat_history", [])]

    if not question:
        return {"error": "Question is required."}

    # ------------------------------
    # 1. Initialize or Fetch Session
    # ------------------------------
    if not chat_history:
        session_id = str(uuid4())  # Generate a unique session ID for new sessions
        logging.info(f"Generated new session ID: {session_id}")
        chat_history = [{"user": f"Session ID: {session_id}", "ai": "Welcome! How can I assist you?"}]

        # Log new session initialization
        update_s3_file(f"New Session Initialized: {session_id}\n{'=' * 50}\n")

    chat_history = get_last_10_conversations()  # Fetch last 10 Q&A pairs from S3

    # ------------------------------
    # 2. Determine Domain
    # ------------------------------
    if not domain:
        if "rcm" in question.lower():
            domain = "RCM"
        elif "dhis" in question.lower():
            domain = "DHIS"
        else:
            domain = None

    if domain and domain.upper() not in SUPPORTED_DOMAINS:
        return {"error": f"Unsupported domain '{domain}'."}

    # ------------------------------
    # 3. Optimize FAQ Matching
    # ------------------------------
    faq_lookup = {qa["question"].lower(): qa["answer"] for qa in faq_data}
    raw_answer = faq_lookup.get(question.lower())
    if raw_answer:
        log_conversation(question, raw_answer)

        # Format FAQ response into HTML
        formatted_response = chat.invoke(HTML_PROMPT_TEMPLATE.format(answer=raw_answer))
        return JSONResponse(content={"response": formatted_response.content})

    # ------------------------------
    # 4. LLM Response Generation Without Streaming
    # ------------------------------
    try:
        # Pass chat history and other params to `run_llm`
        generated_response = run_llm(
            query=question,
            chat=chat,
            docsearch=docsearch,
            chat_history=format_chat_history(chat_history),
            domain=domain
        )

        # Handle different types of `generated_response`
        if isinstance(generated_response, str):
            # If the response is a string, treat it as the full answer
            answer = generated_response

        elif isinstance(generated_response, dict):
            # If the response is a dictionary, extract the "answer" key
            answer = generated_response.get("answer", "")

        elif hasattr(generated_response, "content"):
            # If the response is an object with a "content" attribute
            answer = generated_response.content

        else:
            # Unexpected response type
            raise TypeError(f"Unexpected response type: {type(generated_response)}")

        # Log the conversation for debugging/auditing
        log_conversation(question, answer)

        # Return the full response as JSON
        return JSONResponse(content={"response": answer})

    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the response.")
