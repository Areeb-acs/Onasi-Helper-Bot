from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from backend.core import run_llm
from langchain_groq import ChatGroq
import json
import os

app = FastAPI()
# File to store conversations
CONVERSATION_LOG_FILE = "conversations.txt"
SUPPORTED_DOMAINS = {"RCM", "DHIS"}

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
            # If an exact match is found in FAQ, format the answer in HTML
            raw_answer = qa_pair["answer"]
                        # Log conversation
            log_conversation(question, raw_answer)
            # Use ChatGroq to format the answer in HTML
            formatted_response = chat.invoke(
                HTML_PROMPT_TEMPLATE.format(answer=raw_answer)
            )
                        # Log conversation
            
            # Return HTML response
            return HTMLResponse(content=formatted_response.content)

    # If no FAQ match, proceed with normal processing
    # Determine domain dynamically if not provided
    if not domain:
        if "RCM" in question:
            domain = "RCM"
        elif "DHIS" in question:
            domain = "DHIS"
        else:
            domain = None  # Default to no specific domain

    # Validate the domain if specified
    if domain and domain not in SUPPORTED_DOMAINS:
        return {"error": f"Unsupported domain '{domain}'. Supported domains are: {', '.join(SUPPORTED_DOMAINS)}"}

    async def response_generator():
        generated_response = run_llm(query=question, chat_history=chat_history, domain=domain)
        answer = generated_response.get("answer", "")
        log_conversation(question, answer)
        for chunk in answer:
            yield chunk

    # Log the request for debugging purposes
    print(f"Received query: {question}, Domain: {domain}")

    return StreamingResponse(response_generator(), media_type="text/plain")
