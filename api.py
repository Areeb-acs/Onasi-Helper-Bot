from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from backend.core import run_llm
from langchain_groq import ChatGroq
import json
import os
from dotenv import load_dotenv
import requests
import base64

from uuid import uuid4  # For generating unique session IDs

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

def fetch_file_sha():
    """
    Fetch the SHA of the file on GitHub (required for updates).
    """
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        file_info = response.json()
        return file_info["sha"], base64.b64decode(file_info["content"]).decode("utf-8")
    else:
        print(f"Error fetching file metadata: {response.json()}")
        return None, None

def update_github_file(new_content):
    """
    Update the file on GitHub with the new content.
    """
    sha, current_content = fetch_file_sha()
    if not sha:
        print("Unable to fetch file metadata.")
        return

    updated_content = current_content + new_content
    encoded_content = base64.b64encode(updated_content.encode("utf-8")).decode("utf-8")

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    data = {
        "message": "Update conversations",
        "content": encoded_content,
        "branch": BRANCH,
        "sha": sha
    }

    response = requests.put(url, headers=headers, json=data)
    if response.status_code == 200:
        print("GitHub file updated successfully!")
    else:
        print(f"Error updating GitHub file: {response.json()}")

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
    """
    Reads the last ten interactions from the conversations.txt file on GitHub for the specified session_id.
    """
    file_content = fetch_github_file()
    if not file_content:
        return []

    # Split interactions based on separators
    interactions = file_content.split("=" * 50)
    
    # Filter interactions by session_id
    relevant_interactions = [
        interaction.strip()
        for interaction in interactions
        if f"Session ID: {session_id}" in interaction
    ]

    # Get the last ten non-empty interactions
    last_ten = relevant_interactions[-10:]

    # Parse the last ten interactions into a structured format
    chat_history = []
    for interaction in last_ten:
        user_line = next((line for line in interaction.splitlines() if line.startswith("User:")), None)
        ai_line = next((line for line in interaction.splitlines() if line.startswith("AI:")), None)
        if user_line and ai_line:
            chat_history.append({
                "user": user_line.replace("User: ", "").strip(),
                "ai": ai_line.replace("AI: ", "").strip()
            })

    return chat_history

def log_conversation(session_id, user_query, ai_response):
    """
    Logs the conversation to a local text file, including session_id.
    """
    with open(CONVERSATION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"Session ID: {session_id}\n")
        f.write("User: " + user_query + "\n")
        f.write("AI: " + ai_response + "\n")
        f.write("=" * 50 + "\n")  # Separator for readability

@app.get("/")
async def get_root():
    return {"message": "Welcome to Onasi Helper Bot!"}
@app.post("/chat")
async def chat_endpoint(request: Request):
    # Parse request body
    data = await request.json()
    question = data.get("question")
    session_id = data.get("session_id")  # Client-provided session ID
    domain = data.get("domain", None)

    # Ensure the required parameters are provided
    if not question:
        return {"error": "Question is required."}

    # Check if the session ID is provided; if not, create a new session
    if not session_id:
        session_id = str(uuid4())  # Generate a new session ID

    # Retrieve the last five conversations for the session ID
    chat_history = get_last_ten_conversations(session_id)

    # First, check if there's a direct match in FAQ data
    for qa_pair in faq_data:
        if question.lower() in qa_pair["question"].lower():
            raw_answer = qa_pair["answer"]

            # Log and append the conversation to the session's chat history
            log_conversation(session_id, question, raw_answer)
            chat_history.append({"user": question, "ai": raw_answer})

            # Update GitHub with the new conversation
            update_github_file(f"\nSession ID: {session_id}\nUser: {question}\nAI: {raw_answer}\n{'=' * 50}\n")

            # Format response for HTML and return
            formatted_response = chat.invoke(
                HTML_PROMPT_TEMPLATE.format(answer=raw_answer)
            )
            return {"session_id": session_id, "response": HTMLResponse(content=formatted_response.content)}

    # If no FAQ match, proceed with normal processing
    if not domain:
        if "RCM" in question:
            domain = "RCM"
        elif "DHIS" in question:
            domain = "DHIS"
        else:
            domain = None

    if domain and domain not in SUPPORTED_DOMAINS:
        return {"error": f"Unsupported domain '{domain}'."}

    # Format the session-specific chat history for LLM processing
    formatted_history = format_chat_history(chat_history)

    async def response_generator():
        # Generate response using the run_llm pipeline
        generated_response = run_llm(query=question, chat_history=formatted_history, domain=domain)
        answer = generated_response.get("answer", "")

        # Log the conversation and append it to the session's history
        log_conversation(session_id, question, answer)
        chat_history.append({"user": question, "ai": answer})

        # Update GitHub with the new conversation
        update_github_file(f"\nSession ID: {session_id}\nUser: {question}\nAI: {answer}\n{'=' * 50}\n")

        # Yield chunks of the response for streaming
        for chunk in answer:
            yield chunk

    print(f"Session ID: {session_id}, Received query: {question}, Domain: {domain}")
    return StreamingResponse(response_generator(), media_type="text/plain")