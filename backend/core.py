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

def run_llm(query: str, chat_history, domain=None):
    """
    Main pipeline for processing user queries with priority for FAQ/QA domain
    """
    # Initialize Pinecone with embeddings
    docsearch = Pinecone(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")



    # If a meaningful response is found, return it immediately

    # If no domain is specified, search all documents without a filter
    if not domain:
        domain_retriever = docsearch.as_retriever(
            search_kwargs={
                "filter": {},  # No domain filter
                "k": 5
            }
        )
    else:
        # Use domain-specific search
        domain_retriever = docsearch.as_retriever(
            search_kwargs={
                "filter": {"domain": domain},
                "k": 5
            }
        )
    # Rest of your existing code for domain-specific search
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_template(
        """
        You are a very friendly conversational chatbot that remembers context across a conversation. Use the provided conversation history to understand the user's question and provide clear, concise, and accurate responses for users.
        Only answer based on given context and if context not relevant, please say I do not know. Please give shortest answers possible to questions unless asked otherwise.
        Do not make up answers. Provide direct responses without any explanatory notes or parenthetical comments.
        Never ever share username and passwords.

        Instructions:
        Provide direct responses without any explanatory notes or parenthetical comments.
        Please provide output in html format having bullet points, paragraph breaks, neat bullet points.
        Only answer based on given context.

        1. If there is any NULL character or empty string, then replace that with no information found.
        2.If no relevant response, say I don't know.
        3.Exact exact wording, pick only the most most relevant related response, like be very concise.
        4. Always output the response in html not in plain text
        5. Always refer to the conversation history for context and maintain continuity in your responses but please be direct.
        6. By default, your answers should not be more than 2 sentences, unless user asks for detailed information, if there is no information, say you do not know.

        Context from documents:
        {context}

        Current Query:
        {input}
        """
    )
    
    rephrase_prompt = ChatPromptTemplate.from_template( 
    """
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    Please keep the response in a neat format always using bullet points and breaking down things into sections.

    Chat History:
    {chat_history}

    Follow Up Input: {input}

    Standalone Question:
    """
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm=chat,
        retriever=domain_retriever,
        prompt=rephrase_prompt
    )
    
    result = parameter_based_search(query, docsearch, num_chunks=3)
    additional_context = "\n".join([doc.page_content for doc in result])
    
    stuff_documents_chain = create_stuff_documents_chain(
        chat,
        retrieval_qa_chat_prompt
    )
    
    query_with_context = f"{query}\n\nAdditional Context:\n{additional_context}"
    
    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_documents_chain
    )
    
    result = qa.invoke({
        "input": query_with_context,
        "chat_history": chat_history,
    })
    
    return result


    # If no matches found in QA domain