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




def fetch_query_results(sql_query: str):
    """
    Function to connect to SQL Server, execute a dynamically provided query, and return the results in JSON format.
    
    Args:
        sql_query (str): The SQL query to execute.
    
    Returns:
        str: JSON string containing the query results.
    """


    # Define connection parameters
    server = os.getenv("SERVER_NAME")  # Replace with your server name/IP
    database = os.getenv("DATABASE_NAME")  # Replace with your database name
    username = os.getenv("DB_USERNAME")  # Replace with your username
    password = os.getenv("DB_PASSWORD")  # Replace with your password

    # Create the connection string
    connection_string = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
    )


    try:
        # Establish the connection
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        print("Connection established successfully.")

        # Execute the SQL query
        cursor.execute(sql_query)

        # Fetch all results
        rows = cursor.fetchall()

        # Get column names from the cursor description
        columns = [column[0] for column in cursor.description]

        # Convert rows into a list of dictionaries
        results = [dict(zip(columns, row)) for row in rows]

        # Close the cursor and connection
        cursor.close()
        conn.close()

        # Convert the results to a JSON string
        return json.dumps(results, indent=4)

    except pyodbc.Error as e:
        print("Error in connection or query execution:", e)
        return json.dumps({"error": str(e)})  # Return error as JSON


def generate_sql_query_business_validation(query: str):
    """
    Generate an SQL query based on the user query using a specialized prompt template.
    
    Args:
        query (str): The user query to generate the SQL query for.
        chat_history (list): Conversation history for context (optional).
        
    Returns:
        str: Generated SQL query.
    """
    # Initialize the ChatGroq LLM
    chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

    # SQL generation-specific prompt tuned for the given dataset
    sql_generation_prompt = ChatPromptTemplate.from_template(
        """
        You are an SQL expert working with a dataset named Nphies_Validation_Error_Codes.       
        The table contains the following columns:

        - [Rule_ID]: Integer representing the type ID e.g., BV-00495
        - [Rule_Type]: String representing the category of the type e.g., BV
        - [Rule_Type_Description]: String representing the name of the type e.g., Business and Validation Rule
        - [Rule_Related_Message_Resource_Element]: String describing the type.
        - [Description]: Integer representing the code value.
        - [TMB_Version]: String representing the display value of the code.

        Based on the user's query, generate an accurate SQL query to retrieve the required data.
        Ensure the query is precise and adheres to SQL Server syntax.
        If user asks hello or any irrelevant question, please do not generate any SQL query
        
        Instructions:
        - Only generate SQL queries relevant to the Nphies_Validation_Error_Codes table.
        - If a WHERE condition is required, include it based on the user's query.
        - The CodeValue is sring so always search for string version
        - Do not provide explanations, only return the SQL query.

        User Query:
        {input}
        """
    )


    # Use format_prompt to prepare the prompt
    prompt_value = sql_generation_prompt.format_prompt(input=query)

    # Invoke the LLM to generate the SQL query
    response = chat.invoke(prompt_value.to_messages())

    # Debugging: Print raw response
    print("Raw LLM Response:", response)

    # Extract and sanitize the response content
    if hasattr(response, "content"):
        raw_query = response.content.strip()

        # Remove any wrapping backticks or quotes
        sanitized_query = raw_query.strip("`\"'")

        # Validate that the response contains a valid SQL SELECT statement
        if "select" in sanitized_query.lower() and "from" in sanitized_query.lower():
            return sanitized_query
        else:
            raise ValueError(f"Generated query is not a valid SELECT statement: {sanitized_query}")
    else:
        print(f"Unexpected response type: {type(response)}")
        raise ValueError("Unexpected response format from LLM.")


def generate_sql_query_medical_coding(query: str):
    """
    Generate an SQL query based on the user query using a specialized prompt template.
    
    Args:
        query (str): The user query to generate the SQL query for.
        chat_history (list): Conversation history for context (optional).
        
    Returns:
        str: Generated SQL query. 
    """
    # Initialize the ChatGroq LLM
    chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

    # SQL generation-specific prompt tuned for the given dataset
    sql_generation_prompt = ChatPromptTemplate.from_template(
        """
        You are an SQL expert working with a dataset named Medical_Coding.       
        The table contains the following columns:

        - [TypeId]: Integer representing the type ID.
        - [TypeCategory]: String representing the category of the type.
        - [TypeName]: String representing the name of the type.
        - [Description]: String describing the type.
        - [CodeValue]: Integer representing the code value.
        - [CodeDisplayValue]: String representing the display value of the code.
        - [CodeDefinition]: String providing the definition of the code.
        - [LongDescription]: String providing additional details (can be NULL).

        Based on the user's query, generate an accurate SQL query to retrieve the required data.
        When a user asks what is the codevalue or code value of MRI Scan or anything, it refers to the CodeDisplayValue
        so you will search for all columns where Code Display Value = MRI Scan etc., this is just an example.
        Ensure the query is precise and adheres to SQL Server syntax.
        If user asks hello or any irrelevant question, please do not generate any SQL query
        
        Instructions:
        - Only generate SQL queries relevant to the Medical_Coding table.
        - If a WHERE condition is required, include it based on the user's query.
        - The CodeValue is sring so always search for string version
        - Do not provide explanations, only return the SQL query.

        User Query:
        {input}
        """
    )


    # Use format_prompt to prepare the prompt
    prompt_value = sql_generation_prompt.format_prompt(input=query)

    # Invoke the LLM to generate the SQL query
    response = chat.invoke(prompt_value.to_messages())

    # Debugging: Print raw response
    print("Raw LLM Response:", response)

    # Extract and sanitize the response content
    if hasattr(response, "content"):
        raw_query = response.content.strip()

        # Remove any wrapping backticks or quotes
        sanitized_query = raw_query.strip("`\"'")

        # Validate that the response contains a valid SQL SELECT statement
        if "select" in sanitized_query.lower() and "from" in sanitized_query.lower():
            return sanitized_query
        else:
            raise ValueError(f"Generated query is not a valid SELECT statement: {sanitized_query}")
    else:
        print(f"Unexpected response type: {type(response)}")
        raise ValueError("Unexpected response format from LLM.")



def fetch_code_display_value(code: str):
    """
    Fetch the CodeDisplayValue for a given CodeValue.
    
    Args:
        code (str): The CodeValue to search for.
    
    Returns:
        str: The CodeDisplayValue associated with the CodeValue, or an error message.
    """
    # Define the SQL query
    sql_query = f"SELECT CodeDisplayValue FROM Sys_Codes WHERE CodeValue = {code}"

    # Fetch the query result
    result = fetch_query_results(sql_query)

    # Parse and return the CodeDisplayValue from the result
    try:
        result_json = json.loads(result)
        if result_json and isinstance(result_json, list) and "CodeDisplayValue" in result_json[0]:
            return result_json[0]["CodeDisplayValue"]
        else:
            return f"No CodeDisplayValue found for CodeValue: {code}"
    except json.JSONDecodeError as e:
        return f"Error parsing result: {e}"



