from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from backend.core import run_llm
from langchain_groq import ChatGroq
import json
import os
from dotenv import load_dotenv
import requests
import base64
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


def log_conversation(user_query, ai_response):
    """
    Logs the conversation to a local text file.
    """
    with open(CONVERSATION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write("User: " + user_query + "\n")
        f.write("AI: " + ai_response + "\n")
        f.write("="*50 + "\n")  # Separator for readability

@app.get("/")
async def get_root():
    return {"message": "Welcome to Onasi Helper Bot!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    # Parse request body
    data = await request.json()
    question = data.get("question")
    chat_history = data.get("chat_history", [])
    domain = data.get("domain", None)

    # Ensure the required parameters are provided
    if not question:
        return {"error": "Question is required."}

    # First, check if there's a direct match in FAQ data
    for qa_pair in faq_data:
        if question.lower() in qa_pair["question"].lower():
            raw_answer = qa_pair["answer"]
            log_conversation(question, raw_answer)

            # Update GitHub
            update_github_file(f"\nUser: {question}\nAI: {raw_answer}\n{'='*50}\n")

            # Format response and return
            formatted_response = chat.invoke(
                HTML_PROMPT_TEMPLATE.format(answer=raw_answer)
            )
            return HTMLResponse(content=formatted_response.content)

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

    async def response_generator():
        generated_response = run_llm(query=question, chat_history=chat_history, domain=domain)
        answer = generated_response.get("answer", "")
        log_conversation(question, answer)

        # Update GitHub
        update_github_file(f"\nUser: {question}\nAI: {answer}\n{'='*50}\n")

        for chunk in answer:
            yield chunk

    print(f"Received query: {question}, Domain: {domain}")
    return StreamingResponse(response_generator(), media_type="text/plain")