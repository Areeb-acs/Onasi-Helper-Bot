from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from backend.core import run_llm

app = FastAPI()

@app.get("/")
async def get_root():
    return {"message": "Welcome to Onasi Helper Bot!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    # Parse request body
    data = await request.json()
    question = data.get("question")
    chat_history = data.get("chat_history", [])

    # Ensure the required parameters are provided
    if not question:
        return {"error": "Question is required."}

    async def response_generator():
        # Generate response dynamically using run_llm
        generated_response = run_llm(query=question, chat_history=chat_history)
        answer = generated_response.get("answer", "")
        # Stream the response in chunks
        for chunk in answer:
            yield chunk

    return StreamingResponse(response_generator(), media_type="text/plain")
