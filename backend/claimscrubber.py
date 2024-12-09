from dotenv import load_dotenv
import logging

# https://github.com/emarco177/documentation-helper/blob/2-retrieval-qa-finish/ingestion.py

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




def analyze_fhir_message_and_get_explanation(fhir_content):
    """
    Analyzes the FHIR message and retrieves explanations for error codes.

    Args:
        fhir_content (str): The FHIR message as a string.
        chat_helper: The chatbot function that explains the error codes.

    Returns:
        str: Chatbot's explanation for the analysis result.
    """
    

        
    
    required_keys = ["careTeamSequence", "diagnosisSequence", "informationSequence"]
    missing_keys = []

    for key in required_keys:
        if key not in fhir_content:
            missing_keys.append(key)

    if missing_keys:
        error_messages = []
        for key in missing_keys:
            if key == "careTeamSequence":
                error_messages.append(f"Missing {key}: IC-3898")
            elif key == "diagnosisSequence":
                error_messages.append(f"Missing {key}: IC-3893")
            elif key == "informationSequence":
                error_messages.append(f"Missing {key}: IC-3899")

        analysis_result = "Error: " + ", ".join(error_messages)
    else:
        analysis_result = "Success: All required keys are present."


    return analysis_result

def chat_helper(chat, response):
    """
    Uses a chat model to explain error codes dynamically based on the response.

    Args:
        chat: Pre-initialized chat model (e.g., ChatGroq).
        response (str): The analysis result containing error codes.

    Returns:
        str: Detailed explanation from the chat model.
    """
    # Define a prompt template for error explanation
    prompt_template = ChatPromptTemplate.from_template(
        """
        The following FHIR message analysis result contains error codes. 
        Your task is to explain each error code in detail, providing actionable guidance to address the issue.
        
        <b>Analysis Result:</b> {response}
        
        For each error code:
        - Provide the reason why the error occurred only

        If there are no issues detected, confirm that the analysis result is successful.
        """
    )

    # Format the response into the prompt
    formatted_prompt = prompt_template.format(response=response)

    # Pass the formatted prompt to the chat model
    chat_response = chat.invoke(formatted_prompt)

    return chat_response.content



if __name__ == '__main__':
    # Load FHIR content from the file
    with open('fhir_message.txt', 'r') as fhir_file:
        fhir_content = fhir_file.read()
        print("FHIR message content loaded successfully.")
    
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

    # Analyze the FHIR content
    analysis_result = analyze_fhir_message_and_get_explanation(fhir_content)
    print(analysis_result)
    result = chat_helper(chat, analysis_result)
    # Print the analysis result
    print(result)