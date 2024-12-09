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
from .sql_queries import generate_sql_query_business_validation, generate_sql_query_medical_coding, fetch_query_results
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

    # Initialize context flag
    database_context_provided = False
    database_context = ""



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
        - Please always output response in plain english simple to understand to the user. 

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


    # ---------------------
    # Pattern 1: Business Validation Rules
    # ---------------------
    business_rule_pattern = r"\b(BV|DT|FR|GE|IB|IC|RE)-\d{5}\b|(?i)\b(business validation rule|error|validation error|validation rule)\b"
    if re.search(business_rule_pattern, query):
        try:
            # Generate SQL query for the matching validation rule
            sql_query = generate_sql_query_business_validation(query)  # Replace with actual implementation
            # Fetch results from the database
            results = fetch_query_results(sql_query)  # Replace with actual implementation
            
            if results:
                database_context_provided = True
                database_context = f"Business Validation Rule Results:\n{results}"
                query = f"{query}\n\n{database_context}"
            else:
                return "Sorry, no relevant data found for the specified business validation rule."
        except Exception as e:
            return f"An error occurred while processing the business validation rule: {str(e)}"

    # ---------------------
    # Pattern 2: Medical Codes and Descriptions
    # ---------------------
    medical_code_pattern = r"(?i)\b(Medical Care|Endodontics|Dental Care|codevalue|code value|code display value|description|explain codevalue|explain code)\b"
    code_value_pattern = r"\b(\d+(\.\d+)?|[A-Z]-\w+-\d+)\b"

    if re.search(medical_code_pattern, query) or re.search(code_value_pattern, query):
        try:
            # Generate SQL query for medical codes and descriptions
            sql_query = generate_sql_query_medical_coding(query)  # Replace with actual implementation
            # Fetch results from the database
            results = fetch_query_results(sql_query)  # Replace with actual implementation
            
            if results:
                database_context_provided = True
                database_context = f"Medical Coding Results:\n{results}"
                query = f"{query}\n\n{database_context}"
            else:
                return "Sorry, no relevant data found for the specified medical code or description."
        except Exception as e:
            return f"An error occurred while processing the medical code or description: {str(e)}"

    # ------------------------------
    # If data was retrieved from the database, skip retriever setup
    # ------------------------------
    if database_context_provided:
        formatted_prompt = retrieval_qa_chat_prompt.format(
            context=database_context,
            chat_history=chat_history,
            input=query
        )

        response = chat.invoke(formatted_prompt)  # Generate the response
        return response
    
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
                "k": 4  # Retrieve top 7 results.   
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
