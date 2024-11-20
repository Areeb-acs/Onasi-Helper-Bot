from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from backend.core import run_llm

app = FastAPI()

SUPPORTED_DOMAINS = {"RCM", "DHIS"}

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
        # Generate response using run_llm with domain
        # run_llm will handle the QA check first for RCM domain
        generated_response = run_llm(query=question, chat_history=chat_history, domain=domain)
        answer = generated_response.get("answer", "")
        # Stream the response in chunks
        for chunk in answer:
            yield chunk

    # Log the request for debugging purposes
    print(f"Received query: {question}, Domain: {domain}")

    return StreamingResponse(response_generator(), media_type="text/plain")
