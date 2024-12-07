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
    # is_summary_request = any(keyword in query.lower() for keyword in ["summarize", "summarise", "reword", "parahparse"])
    is_summary_request = False
    if is_summary_request:
            # Create the system message

        # Create the prompt template
        # summary_prompt = ChatPromptTemplate.from_template(
        #     """
        #     You are a helpful assistant. The user has asked for a summarization or rewording of the latest context.
        #     Use ONLY the provided chat history to create your response. Do not include additional information.
        #     Please output a summary in your own words neatly summarizing the AI response only.
        #     Please only use the last line to use as information only.

        #     ALWAYS OUTPUT in <html> elements but never use the <html> tag itself.
        #     <b>Instructions:</b>
        #     - Summarize or reword the provided context as requested.
        #     - Use clear and concise language.
        #     - Respond with bullet points if necessary but do not include additional explanations.
        #     - Create sub-bullet points as well using nested <ul> tags for better readability.
        #     - There is a line break after each bullet point for better readability.
        #     - Avoid markdown; always format output in clean HTML (no <html> tag).
            
        #     - If the context is irrelevant or insufficient, reply with "I don't know."
        #     - Never hallucinate information; use the provided chat history only.
        #     - Respond with detailed explanations when required but always concise.
        #     - Respond with bullet points when the answer is longer than 2 sentences.

        #     <b>Current Query:</b> {input}
        #     <b>Conversation History:</b> {chat_history}
        #     """
        # )
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
                "k": 3  # Retrieve top 7 results.
            }
        )
    else:
        # Use domain-specific filtering for more focused results.
        domain_retriever = docsearch.as_retriever(
            search_kwargs={
                "filter": {"domain": domain},  # Apply domain-specific filter.
                "k": 3  # Retrieve top 10 results.
            }
        )
        

    # ------------------------------
    # 2. Define Chat Prompts
    # ------------------------------
    # Prompt for contextual retrieval-based QA.
    
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_template(
        """
        Your name is Onasi AI. You are a friendly chatbot that provides concise and accurate responses based solely on the given context.
        Please always give a simplified, user-friendly, and condensed answer unless explicitly asked to be more specific or detailed.

        <b>Rules to Follow:</b>
        1. ALWAYS OUTPUT IN HTML TAGS and use bullet points for medium to long answers. NEVER USE THE <html> TAG ITSELF. NEVER USE MARKDOWN.  
        2. Use nested <ul> and <li> tags for sub-bullet points to enhance readability.
        3. For answers longer than two sentences, break them down into bullet points using HTML tags.
        4. If the answer can be provided in 2-3 words, avoid using bullet points.
        5. If you cannot find the answer in the given context or chat history, say: "Sorry, I cannot help with your query."
        6. When the user greets you (e.g., says "Hello"), respond briefly and do not elaborate further.
        7. Always base your responses on the given context and chat history. Do not generate unrelated or speculative answers.
        8. Never hallucinate information. If the information is not available in the context, respond with "Sorry, I cannot help with your query."
        9. Do not repeat the user's question in your response. Provide a direct and clear answer.
        
        <b>Further Instructions:</b>
        - If the user asks to summarize or reword content, rely solely on the provided context and chat history.
        - Avoid going out of context when summarizing or rewording.
        - If the user's query is irrelevant to the context, say: "Sorry, I cannot help with your query."

        <b>Main Instructions:</b>
        - Use bullet points (<ul> and <li>) to structure responses longer than two sentences.
        - Ensure a line break for better readability after each bullet point.
        - Avoid Markdown formatting; only use clean HTML (without the <html> tag).
        - When the context is insufficient or irrelevant, reply: "Sorry, I cannot help with your query."
        - Ensure that all answers are concise and related strictly to the context provided.
        - Do not prepend "Response:" to your answers. Start directly with the response content.
        - Always avoid long-winded explanations. Be concise, relevant, and user-focused.



        <b>Context:</b> {context}

        <b>Chat History:</b> {chat_history}
        
        
        <b>Current Query:</b> {input}
        """
    )

    # Prompt for rephrasing follow-up questions into standalone queries.
    rephrase_prompt = ChatPromptTemplate.from_template(
        """
        Rephrase the follow-up query to make it a standalone question, considering the conversation history.
        Consider the information and design the question specifically.
        <b>Chat History:</b> {chat_history}


        Follow Up Input: {input}

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
    from langchain_core.output_parsers import StrOutputParser

    # ------------------------------
    # 4. Document Combination Chain
    # ------------------------------
    # Combine retrieved documents for contextual QA.
    stuff_documents_chain = (
        retrieval_qa_chat_prompt
        | chat
        | StrOutputParser()
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
