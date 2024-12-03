from dotenv import load_dotenv


# https://github.com/emarco177/documentation-helper/blob/2-retrieval-qa-finish/ingestion.py
load_dotenv()
import re
import os

from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import Pinecone
from langchain_groq import ChatGroq

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json
import pyodbc
import json
import os
import os
import pyodbc




INDEX_NAME = "rcm-final-app"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

groq_api_key = os.getenv("GROQ_API_KEY")
# Initialize Pinecone

def is_conversation_start(chat_history):
    """
    Check if this is the start of a conversation
    """
    return not chat_history or len(chat_history) == 0

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


def run_llm(query: str, chat_history, chat, docsearch, domain=None):
    """
    Main pipeline for processing user queries with a focus on specific instructions 
    like summarization and rewording.
    
    Parameters:
        query (str): User query input.
        chat_history (list): List of previous conversation turns (user and AI responses).
        chat: Pre-initialized chat model.
        docsearch: Pre-initialized Pinecone object for document retrieval.
        domain (str): Optional domain filter for focused document retrieval.

    Returns:
        str: AI-generated response based on the query and context.
    """
    # ------------------------------
    # 1. Detect Summarize or Reword Requests
    # ------------------------------
    is_summary_request = "summarize" in query.lower() or "reword" in query.lower()
    latest_context = ""
    
    if is_summary_request and chat_history:
        # Use only the latest Q&A pair from the chat history
        latest_entry = chat_history[-2]
        latest_context = f"User: {latest_entry['user']}\nAI: {latest_entry['ai']}\n"

    # ------------------------------
    # 2. Setup Document Retriever
    # ------------------------------
    # Use default retriever if no domain is specified
    if not domain:
        domain_retriever = docsearch.as_retriever(
            search_kwargs={"filter": {}, "k": 5}  # Top 5 results
        )
    else:
        # Domain-specific filtering
        domain_retriever = docsearch.as_retriever(
            search_kwargs={"filter": {"domain": domain}, "k": 5}
        )

    # ------------------------------
    # 3. Define Chat Prompts
    # ------------------------------
    if is_summary_request:
        # Prompt specifically for summarization or rewording
        retrieval_qa_chat_prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant. The user has asked for a summarization or rewording of the latest context.
            Use ONLY the provided context to create your response. Do not include additional information.

            <b>Instructions:</b>
            - Summarize or reword the provided context as requested.
            - Use clear and concise language.
            - Respond with bullet points if necessary but do not include additional explanations.

            <b>Latest Context:</b>
            {chat_history}

            <b>Current Query:</b>
            {input}
            """
        )
    else:
        retrieval_qa_chat_prompt = ChatPromptTemplate.from_template(
            """
            You are a friendly chatbot that provides concise and accurate responses.
            Use the provided conversation history to understand the user's query and answer based on the context.
            Your name is Onasi AI, a friendly conversational chatbot. Only answer based on the provided context.
            If answer is not in given context, please respond I don't know only.
            Do not mention step numbers, the numbering is only for the order. 
            If user just enters vague statements like Good, just answer please ask a valid question.
            No need to start response with bullet point but then you eventually need to provide bullet points.
            
            

            <b>Instructions:</b>
            - WHEN User mentions summarize, user mentions reword the above, for this please use only chat history and the latest information
            - 
            
            Please do not use context in case of these statements, just reply as quickly as possible saying I don''t know.        
            <b>Instructions:</b>
            - Break down your response into bullet points using HTML tags and always format them nicely
            - Create sub-bullet points as well using nested <ul> tags for bette readability
            - There is a link break after each bullet point for better readability
            - Avoid markdown; always format output in clean HTML (no <html> tag).
            - If the context is irrelevant or insufficient, reply with "I don't know."        
            - If answer is not in given context, please respond I don't know.
            - Never hallucinate information; use the provided context only.
            - Respond with detailed explanations when required but always concise.
            - Respond with bullet points when answer is longer than 2 sentences.
            - Please do not use Markdown, only HTML tags for bullet point formatting only <ul> and <li> elements
            - Please do not start with Response <b>Response:</b>, directly answer the question.
            - If answer is very short, no need for bullet points.
            
            
            <b>Current Query:</b> {input}
            <b>Context:</b> {context}

            <b>Conversation History:</b> {chat_history}
            """
        )

    # Prompt for rephrasing follow-up questions into standalone queries.
    rephrase_prompt = ChatPromptTemplate.from_template(
        """
        Rephrase the follow-up query to make it a standalone question, considering the conversation history.

        Follow-Up Input: {input}

        Standalone Question:
        """
    )

    # ------------------------------
    # 3. History-Aware Retriever
    # ------------------------------
    history_aware_retriever = create_history_aware_retriever(
        llm=chat,  # Use the pre-initialized chat model.
        retriever=domain_retriever,  # Use the document retriever.
        prompt=rephrase_prompt  # Rephrase follow-up questions when needed.
    )

    # ------------------------------
    # 4. Document Combination Chain
    # ------------------------------
    # Combine retrieved documents for contextual QA.
    stuff_documents_chain = create_stuff_documents_chain(
        chat,  # Use the same chat model for response generation.
        retrieval_qa_chat_prompt  # Use the retrieval-based QA prompt.
    )

    # ------------------------------
    # 5. QA Chain Execution
    # ------------------------------
    qa_chain = create_retrieval_chain(
        retriever=history_aware_retriever,  # Context-aware retriever.
        combine_docs_chain=stuff_documents_chain  # Combine documents for the final answer.
    )

    # Run the QA chain with the query and formatted chat history.
    result = qa_chain.invoke({
        "input": query,  # User query.
        "chat_history": chat_history  # Format chat history for input.
    })

    # Return the generated response.
    return result
