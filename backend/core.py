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
docsearch = Pinecone(index_name=INDEX_NAME, embedding=embeddings)

def extract_numbers_from_query(query):
    """
    Extract numbers from the query using regex.
    """
    return list(map(int, re.findall(r'\b\d+\b', query)))

def get_all_documents(vector_store, domain=None):
    """
    Retrieve all documents from the Pinecone vector store.
    """


    if domain:
        retriever = vector_store.as_retriever(search_kwargs={"filter": {"domain": domain}})
        all_docs = retriever.get_relevant_documents("")  # Retrieve all documents
    else:
        # Fall back to embedding-based search for more complex queries
        retriever = vector_store.as_retriever()
        all_docs = retriever.get_relevant_documents("")
    

    return all_docs  # Ensure these are Document objects




def parameter_based_search(query, vector_store, num_chunks=15, file_type=None, domain=None):
    """
    Search documents based on parameters:
    1. Searches `coding.json` for `CodeValue` or `Code Value` matches.
    2. Searches `validation.json` for `Rule ID` matches (format: two letters followed by numbers).
    Falls back to embedding-based search for more complex queries.
    """
    import re

    # Check if CodeValue or Code Value (with or without space) is mentioned in the query
    is_code_value_search = re.search(r"\bCode\s?Value\b", query, re.IGNORECASE)

    # Extract numbers in the query
    numbers_in_query = re.findall(r"\b\d+\b", query)

    # Extract Rule ID from the query
    rule_id_match = re.search(r"\b[A-Za-z]{2}-\d+\b", query)  # Match "XX-12345" format Rule IDs

    matches = []

    # Retrieve all documents from the vector store
    final_documents = get_all_documents(vector_store, domain=domain)

    # If numbers are present and CodeValue (or Code Value) is explicitly mentioned
   # If numbers are present and CodeValue (or Code Value) is explicitly mentioned
    if is_code_value_search and numbers_in_query:
        for number in numbers_in_query:
            number_str = str(number).strip()  # Convert the number to a string
            quoted_number = f'"{number_str}"'  # Add double quotes if needed for strict matching
            print(f"Searching for CodeValue: {number_str} (and as quoted: {quoted_number})")  # Debugging output

            # Use retriever with direct filter for CodeValue
            retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": num_chunks,
                    "filter": {"file_type": "coding", "CodeValue": number_str}
                }
            )

            # Retrieve documents
            filtered_docs = retriever.get_relevant_documents(query)

            if filtered_docs:
                print(f"Found {len(filtered_docs)} matches for CodeValue: {number_str}")
                matches.extend(filtered_docs[:num_chunks])
            else:
                print(f"No matches found for CodeValue: {number_str}.")

    # Check for Rule ID matches
    if rule_id_match:
        rule_id_to_search = rule_id_match.group(0).strip()  # Extract the Rule ID
        print(f"Searching for Rule ID: {rule_id_to_search}")  # Debugging output

        # Use retriever with direct filter for Rule ID
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": num_chunks,
                "filter": {"file_type": "validation", "Rule ID": rule_id_to_search}
            }
        )

        # Retrieve documents
        filtered_docs = retriever.get_relevant_documents(query)

        if filtered_docs:
            print(f"Found {len(filtered_docs)} matches for Rule ID: {rule_id_to_search}")
            matches.extend(filtered_docs[:num_chunks])
        else:
            print(f"No matches found for Rule ID: {rule_id_to_search}.")


    # Fall back to embedding-based search for more complex queries
    if not matches:
        print("No exact matches found. Falling back to embedding-based search.")  # Debugging output
        if domain:
            retriever = vector_store.as_retriever(search_kwargs={"k": num_chunks, "filter": {"domain": domain}})
        else:
            retriever = vector_store.as_retriever(search_kwargs={"k": num_chunks})

        similar_docs = retriever.get_relevant_documents(query)
        embedding_matches = [doc for doc in similar_docs if hasattr(doc, "page_content")]
        matches.extend(embedding_matches)

    # Return the matches
    print(f"Found {len(matches)} matches.")  # Debugging output
    return matches

def is_conversation_start(chat_history):
    """
    Check if this is the start of a conversation
    """
    return not chat_history or len(chat_history) == 0


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
    # username = os.getenv("DB_USERNAME")  # Replace with your username
    # password = os.getenv("DB_PASSWORD")  # Replace with your password

    # Create the connection string
    # connection_string = (
    #     f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    #     f"SERVER={server};"
    #     f"DATABASE={database};"
    #     # f"UID={username};"
    #     # f"PWD={password};"
    # )

    connection_string = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Trusted_Connection=yes;"
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


def generate_sql_query(query: str, chat_history=None):
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
        You are an SQL expert working with a dataset named RCM_dataset. First check the query, see if it related to any columns below, if so only then generate a sql query  
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
        Ensure the query is precise and adheres to SQL Server syntax.
        
        Instructions:
        - Only generate SQL queries relevant to the RCM_dataset table.
        - If a WHERE condition is required, include it based on the user's query.
        - If you determine no need for a sql query, please output None     
        - Do not provide explanations, only return the SQL query.

        User Query:
        {input}
        """
    )

    # Prepare the input for the LLM
    input_with_context = query
    if chat_history:
        input_with_context += f"\n\nChat History:\n{chat_history}"

    # Use format_prompt to prepare the prompt
    prompt_value = sql_generation_prompt.format_prompt(input=input_with_context)

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
            return "None"
            # raise ValueError(f"Generated query is not a valid SELECT statement: {sanitized_query}")
    else:
        print(f"Unexpected response type: {type(response)}")
        return "None"
        # raise ValueError("Unexpected response format from LLM.")

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
    Main pipeline for processing user queries with priority for FAQ / QA domain
    """

    
    # If no domain is specified, search all documents without a filter
    if not domain:
        domain_retriever = docsearch.as_retriever(
            search_kwargs={
                "filter": {},  # No domain filter
                "k": 7
            }
        )

    else:
        # Use domain-specific search
        domain_retriever = docsearch.as_retriever(
            search_kwargs={
                "filter": {"domain": domain},
                "k": 10
            }
        )

    
    # Rest of your existing code for domain-specific search
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_template(
        """
        
        You are a very friendly conversational chatbot that remembers context across a conversation. Use the provided conversation history to understand the user's question and provide clear, concise, and accurate responses for users.
        Only answer based on given context and if context not relevant, please say I do not know. Give short answers but when detaile are needed, please give an elaborate answer in bullet points.
        Always use the context for information only, but do reword and rephrase for user to understand complex explanations. 
        Do not make up answers. Provide direct responses without any explanatory notes or parenthetical comments. 
        
        When you cannot find an answer, please say this is out of scope for me or out of my knowledge base.
        
        BASIC RULE: ALWAYS BREAKDOWN YOUR ANSWER IN BULLET POINTS WHEN GIVING STEP BY STEP EXPLANATIONS AND OUTPUT IN HTML TAGS. ALWAYS BULLET, NO MARKDOWN PLEASE.
        ALWAYS ANSWER IN BULLET POINTS using HTML tags

        When using terms like RC, RD, RS, do not confuse these with RCM and don't output the definition of RCM in response.
        
        For codevalue and business validation rules, always refer to Additional Context, if no information there, say I don't know.
        For lengthy responses, please provide response in bullet points.
        
        Never ever share username and passwords. Also this is your key responsibility:

            - If a users says hello, no need for bullet points
            - Don't hallucinate please. If the user tells his or her name, reply appropriately. If codevalue not in context and user asked about it, say I do not know.
            - P&S stands for Product and Services.
            - If you cannot find any relevant answer, just say I don't know.
            - Never say that you will output result in html, never tell the user. You are direct, to the point, anything that the user does not need to know,
            don't mention.

        Instructions:
        Provide direct responses without any explanatory notes or parenthetical comments.
        Please provide output using html tags having bullet points, paragraph breaks, neat bullet points but DO NOT put the <html> tag at the start, just other tags.
        ALWAYS ANSWER IN BULLET POINTS using HTML tags

        1. If there is any NULL character or empty string, then replace that with no information found.
        2. If no relevant response, say I don't know.
        3. Always output the response in html not in plain text
        4. Always refer to the conversation history for context and maintain continuity in your responses but please be direct.
        5. Always always breakdown medium to long answers into bullet points nicely formatted in HTML.
                ALWAYS ANSWER IN BULLET POINTS using HTML tags but dont start with <html> tag, just other tags.




    BASIC RULE: ALWAYS OUTPUT IN HTML, ALWAYS, REGARDLESS OF CONVERSATIONAL HISTORY AND CONTEXT. FOR BOLD WORDS starting with **, use the <b> tag instead. PLEASE AVOID MARKDOWN.
    BASIC RULE: ONLY ANSWER BASED ON PROVIDED CONTEXT.
    BASIC RULE: ALWAYS ALWAYS TAKE THE OUTPUT AND FORMAT IT NICELY IN HTML TAGS, replace the '-' with bullet points, MAKE SURE ALL SPACES ARE DISTRIBUTED AND FORMATTED NICELY.
    If the user asks about their name, infer it from the conversation history. If their name was mentioned in the conversation history, respond with their name. If their name was not mentioned, respond politely that you don't know their name.
        ALWAYS ANSWER IN BULLET POINTS using HTML tags

        <b>Context:</b>
        {context} Also please note that pre-authorization is part of the claim process if query is related to claim submission.
        ALWAYS ANSWER IN BULLET POINTS using HTML tags
  

        <b>Current Query:</b>
        {input}

        <b>Conversation History:</b>
        {chat_history}

        """
    )
    
    
    # Define a rephrasing prompt to transform follow-up questions into standalone questions.
    rephrase_prompt = ChatPromptTemplate.from_template(
        """
        Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
        Please keep the response in a neat format always using bullet points and breaking down things into sections.
        ALWAYS ANSWER IN BULLET POINTS WHEN ANSWER IS MORE THAN 2 SENTENCES


        Follow Up Input: {input}

        Standalone Question:
        """
    )

    # Create a history-aware retriever that combines the LLM, retriever, and rephrasing prompt.
    # This ensures that user queries consider the context of the entire chat history.
    history_aware_retriever = create_history_aware_retriever(
        llm=chat,  # The language model to process queries
        retriever=domain_retriever,  # Retrieves relevant documents based on the query
        prompt=rephrase_prompt  # Prompt used for rephrasing follow-up questions
    )

    # Commented out code for skipping parameter-based search on trivial queries like "hello".
    # If the query contains "hello", the context is skipped; otherwise, additional context is fetched.
    # if "hello" in query.lower():
    #     additional_context = ""
    # else:
    #     result = parameter_based_search(query, docsearch, num_chunks=3)
    #     additional_context = "\n".join([doc.page_content for doc in result])

    # Dynamically generate an SQL query based on the user's input.
    # sql_query = generate_sql_query(query)

    # Debugging: Print the generated SQL query to verify its structure.
    # print(sql_query)

    # Validate the SQL query before execution.
    # Only proceed with database interaction if the query is valid and starts with "SELECT".
    # if sql_query and sql_query.strip().lower().startswith("select"):
    #     # Fetch results from the database for valid SQL queries.
    #     results = fetch_query_results(sql_query)
    # else:
    #     # Handle invalid or None queries by providing a default response.
    #     results = 'No Additional Context Found'
    #     # Log skipped execution for debugging purposes.
    #     print("Skipped execution: Invalid SQL query generated or query is None.")

    # Combine the original user query with the additional context (e.g., database results).
    # query_with_context = f"{query}\n\nAdditional Context:\n{results}"

    # Create a document chain for retrieval-based question-answering.
    # This chain combines the LLM and the defined QA prompt to process queries effectively.
    stuff_documents_chain = create_stuff_documents_chain(
        chat,  # The language model used for generating responses
        retrieval_qa_chat_prompt  # Prompt for retrieval-based QA
    )


    # Create a QA chain that integrates the history-aware retriever and the document chain.
    # This chain processes user queries in light of conversation history and retrieved documents.
    qa = create_retrieval_chain(
        retriever=history_aware_retriever,  # Retrieves context-aware responses
        combine_docs_chain=stuff_documents_chain  # Combines document context into the response
    )

    # Execute the QA chain with the provided query and chat history.
    result = qa.invoke({
        "input": query,  # Input query enriched with additional context
        "chat_history": chat_history  # Correctly pass the formatted history
    })

    # Return the result generated by the QA chain.
    return result


    # If no matches found in QA domain