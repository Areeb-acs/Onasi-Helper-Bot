from dotenv import load_dotenv
import logging

# https://github.com/emarco177/documentation-helper/blob/2-retrieval-qa-finish/ingestion.py
load_dotenv()
import re
import os
from langchain_core.messages import SystemMessage, HumanMessage

from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
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
    Main pipeline for processing user queries with priority for FAQ / QA domain.

    Parameters:
        query (str): User query input.
        chat_history (list): List of previous conversation turns (user and AI responses).
        chat: Pre-initialized chat model (e.g., ChatGroq or ChatOpenAI).
        docsearch: Pre-initialized Pinecone object for document retrieval.
        domain (str): Optional domain filter for focused document retrieval.

    Returns:
        str: AI-generated response based on the query and context.
    """

    import time

    # Start total timer
    total_start = time.time()
            # ------------------------------
    embedding_start = time.time()
    # Detect Summarization or Reword Requests
    is_summary_request = any(keyword in query.lower() for keyword in ["summarize", "summarise", "reword", "parahparse"])

    if is_summary_request:
            # Create the system message

        # Create the prompt template
        summary_prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant. The user has asked for a summarization or rewording of the latest context.
            Use ONLY the provided chat history to create your response. Do not include additional information.
            Please output a summary in your own words neatly summarizing the conversational history.
            Please only use the last line to use as information only.

            ALWAYS OUTPUT in <html> elements but never use the <html> tag itself.
            <b>Instructions:</b>
            - Summarize or reword the provided context as requested.
            - Use clear and concise language.
            - Respond with bullet points if necessary but do not include additional explanations.
            - Create sub-bullet points as well using nested <ul> tags for better readability.
            - There is a line break after each bullet point for better readability.
            - Avoid markdown; always format output in clean HTML (no <html> tag).
            
            - If the context is irrelevant or insufficient, reply with "I don't know."
            - Never hallucinate information; use the provided chat history only.
            - Respond with detailed explanations when required but always concise.
            - Respond with bullet points when the answer is longer than 2 sentences.

            <b>Current Query:</b> {input}
            <b>Conversation History:</b> {chat_history}
            """
        )
        # ------------------------------
        # 1. Embedding Retrieval

        # Format the prompt with the input query and chat history
        formatted_prompt = summary_prompt.format(
            input=query,
            chat_history=chat_history  # Ensure this is properly formatted text
        )

        try:
            # Call the ChatGroq API
            response = chat.invoke(formatted_prompt)  # Ensure correct method for the library being used

            # Extract the content from the response object
            if hasattr(response, "content"):
                result = response.content  # Access content attribute (adjust if needed)
            elif hasattr(response, "text"):
                result = response.text  # Alternative access point if applicable
            else:
                raise AttributeError("Response object does not have 'content' or 'text' attributes.")

            return result  # Return the extracted result

        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "An error occurred while generating the response."


    # ------------------------------
    # Proceed with the Standard QA Flow
    # ------------------------------
    # ------------------------------
    # 1. Setup Document Retriever
    # ------------------------------
    # Use a default retriever if no domain is specified.
    if not domain:
        domain_retriever = docsearch.as_retriever(
            search_kwargs={
                "filter": {},  # No filter applied for global search.
                "k": 6  # Retrieve top 7 results.
            }
        )
    else:
        # Use domain-specific filtering for more focused results.
        domain_retriever = docsearch.as_retriever(
            search_kwargs={
                "filter": {"domain": domain},  # Apply domain-specific filter.
                "k": 10  # Retrieve top 10 results.
            }
        )
        

    # ------------------------------
    # 2. Define Chat Prompts
    # ------------------------------
    # Prompt for contextual retrieval-based QA.
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_template(
        """
        
        RULE:
        ALWAYS OUTPUT IN HTML TAGS and use bullet points for medium to long answers, NEVER USE THE <html> TAG ITSELF. NEVER USE MARKDOWN.  
        Create sub-bullet points as well using nested <ul> tags for better readability. If response will be 2-3 words, no need to use bullet points please.
        Please if response is more than 2 sentences, then use bullet points shown in html tags.
        
        If the user asks what is my name or any other question like this, look in conversation history section. 
        If you cannot find the answer to any question in context and chat history, please say I don't know only.
        
        
        When the user says Hello or gives greeting, just simply reply to the greeting, please do not say more than that.
        
        Your name is Onasi AI, You are a friendly chatbot that provides concise and accurate responses based on given context only.
        Use the provided conversation history to understand the user's query and answer based on the context only.
        
        If the user asks a question, please see if the question has already been answered in chat history and respond accordingly
        Learn to say I do not know. This is important to avoid hallucinations. Please answer directly, do not include Response: in start of your response.
        
        Do not mention step numbers, the numbering is only for the order. 
        No need to start response with bullet point but then you eventually need to provide bullet points.
    
        Please do not repeat what the user asked, just answer the question directly. Understand the question.
        <b>Instructions:</b>
        - WHEN User mentions summarize, user mentions reword the above, for this please use only chat history and the latest information
        - If user asks summarize, then please just look at context and conversational history and neatly summarize, do not go out of context.
        - If question is not in given context, please respond I don't know or I am not aware of this or out of my knowledge base.

        
        Please do not use context in case of these statements, just reply as quickly as possible saying I don''t know.        
        <b>Instructions:</b>
        - Break down your response into bullet points using HTML tags and always format them nicely
        - Create sub-bullet points as well using nested <ul> tags for better readability
        - There is a link break after each bullet point for better readability
        - Avoid markdown; always format output in clean HTML (no <html> tag).
        - If the context is irrelevant or insufficient, reply with "I don't know."        
        - If answer is not in given context, please respond I don't know.
        - Never hallucinate information; use the provided context only.
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
        Consider the information and design the question specifically.
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
