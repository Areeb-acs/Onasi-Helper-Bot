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

import time


# def get_last_10_conversations():
#     """
#     Fetches the last 10 Q&A pairs from the S3 conversations file.

#     Returns:
#         List[dict]: A list of the last 10 conversations in the format:
#                     [{"user": "question1", "ai": "answer1"}, ...]
#     """
#     try:
#         # Fetch the content of the file from the S3 bucket
#         response = requests.get(FILE_URL)
#         if response.status_code == 200:
#             content = response.text
#         else:
#             logging.error(f"Failed to fetch conversation file: {response.status_code}")
#             return []

#         # Split content into individual conversations by the separator
#         entries = content.strip().split("==================================================")
#         conversations = []

#         for entry in entries:
#             # Process each entry to extract User and AI lines
#             lines = entry.strip().split("\n")
#             user_line = next((line.replace("User: ", "").strip() for line in lines if line.startswith("User:")), None)
#             ai_line = next((line.replace("AI: ", "").strip() for line in lines if line.startswith("AI:")), None)

#             # Append only valid entries with both User and AI content
#             if user_line and ai_line:
#                 conversations.append({"user": user_line, "ai": ai_line})

#         # Return the last 10 conversations
#         return conversations[-1:] if len(conversations) > 1 else conversations

#     except Exception as e:
#         logging.error(f"Error fetching or parsing conversation file: {str(e)}")
#         return []

start_time = time.time()
print(f"QA chain execution took: {time.time() - start_time:.2f} seconds")

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






def get_conversation_by_session_id(session_id):
    """
    Fetches all user queries for a specific session ID and returns them in a structured format.

    Args:
        session_id (str): The session ID to filter conversations by.

    Returns:
        List[dict]: A list of conversations in the format:
                    [{"user": "question1"}, ...]
    """
    try:
        # Fetch the content of the file from the S3 bucket
        response = requests.get(FILE_URL)
        if response.status_code != 200:
            logging.error(f"Failed to fetch conversation file: {response.status_code}")
            return []

        content = response.text

        # Check if the session ID exists
        session_marker = f"New Session Initialized: {session_id}"
        if session_marker not in content:
            logging.info(f"Session ID {session_id} not found.")
            update_s3_file(f"New Session Initialized: {session_id}\n{'=' * 50}\n")
            return []

        # Split content into lines and process from the session marker onward
        lines = content.splitlines()
        conversations = []
        in_session = False

        for line in lines:
            line = line.strip()

            if line.startswith("New Session Initialized:"):
                # Check if we reached the target session marker
                if line == session_marker:
                    in_session = True  # Start capturing this session
                elif in_session:
                    break  # Exit when the next session starts

            elif in_session:
                # Capture only user lines within the session
                if line.startswith("User:"):
                    user_query = line.replace("User: ", "").strip()
                    conversations.append({"user": user_query})

        return conversations

    except Exception as e:
        logging.error(f"Error fetching or parsing conversation file: {str(e)}")
        return []

def log_conversation(user_query, ai_response):
    """Logs the conversation to the public S3 file."""
    new_entry = f"User: {user_query}\nAI: {ai_response}\n\n"
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
        # ai_response = entry.get("ai", "Unknown response")
        # formatted_history += f"User: {user_input}\nAI: {ai_response}\n"
        formatted_history += f"User: {user_input}\n"
    
    return formatted_history.strip()

@app.get("/")
async def get_root():
    return {"message": "Welcome to Onasi Helper Bot!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    Chat endpoint for processing user queries and returning AI-generated responses.

    Workflow:
    1. Parse incoming request data.
    2. Check for an existing session or initialize a new one.
    3. Match the query against FAQ data for faster responses.
    4. If not found in FAQ, proceed with LLM response generation.
    5. Log the conversation to S3 (batched for efficiency).
    6. Stream the response back to the user.
    """

   

    data = await request.json()
    # print(data)
    question = data.get("question")
    session_id = data.get("session_id")
    domain = data.get("domain", None)
    chat_history = get_conversation_by_session_id(session_id)
    print(format_chat_history(chat_history))

    if not question:
        return {"error": "Question is required."}

    # ------------------------------
    # 1. Initialize or Fetch Session
    # ------------------------------
    if not chat_history:
        logging.info(f"Generated new session ID: {session_id}")

        # Log new session initialization
        

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
    # Use a dictionary for O(1) lookup instead of iterating through the list
    faq_lookup = {qa["question"].lower(): qa["answer"] for qa in faq_data}
    raw_answer = faq_lookup.get(question.lower())
    if raw_answer:

        # Format FAQ response into HTML
        formatted_response = chat.invoke(HTML_PROMPT_TEMPLATE.format(answer=raw_answer))
        return {
            "response": HTMLResponse(content=formatted_response.content)
        }

    # ------------------------------
    # 4. LLM Response Generation
    # ------------------------------
    async def response_generator():
        """
        Generate the response using LLM with streaming.
        """
        # conversation_data = get_last_10_conversations()  # Fetch last 10 Q&A pairs from S3
        import time

        start_time = time.time()
        try:
            # Pass chat history and other params to `run_llm`
            generated_response = run_llm(
                query=question,
                chat=chat,
                docsearch=docsearch,
                chat_history=format_chat_history(chat_history),
                domain=domain
            )
                
            print(f"QA chain execution took: {time.time() - start_time:.2f} seconds")


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

            # Stream response chunks
            for chunk in answer:
                yield chunk

        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            yield "An error occurred while generating the response."


    logging.info(f"Processing query: {question}, Domain: {domain}")
    return StreamingResponse(response_generator(), media_type="text/plain")

