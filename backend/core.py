from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate

# https://github.com/emarco177/documentation-helper/blob/2-retrieval-qa-finish/ingestion.py
load_dotenv()
import re
import os

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import Pinecone
from langchain_groq import ChatGroq

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


INDEX_NAME = "rcm-final-app"
embeddings = OpenAIEmbeddings()

groq_api_key = os.getenv("GROQ_API_KEY")
# Initialize Pinecone
docsearch = Pinecone(index_name=INDEX_NAME, embedding=embeddings)

def extract_numbers_from_query(query):
    """
    Extract numbers from the query using regex.
    """
    return list(map(int, re.findall(r'\b\d+\b', query)))

def get_all_documents(vector_store):
    """
    Retrieve all documents from the Pinecone vector store.
    """
    retriever = vector_store.as_retriever()
    all_docs = retriever.get_relevant_documents("")  # Retrieve all documents
    return all_docs  # Ensure these are Document objects

def rule_based_search(query, vector_store, num_chunks=10, file_type=None):
    """
    Search documents for exact matches of:
    1. Rule IDs (format: BV-XXXXX)
    2. Numbers found in the query
    Returns up to num_chunks matching documents
    Falls back to embedding-based search if no exact matches found
    """
    # Extract numbers or rule ID from the query
    numbers_in_query = extract_numbers_from_query(query)
    rule_id_match = re.search(r"BV-\d{5}", query)  # Adjust regex for your Rule ID format
    print(rule_id_match)

    # Retrieve all documents from the vector store
    final_documents = get_all_documents(vector_store)
    matches = []

    # Check for Rule ID match
    if rule_id_match:
        rule_id_to_search = rule_id_match.group(0)  # Extract matched Rule ID
        rule_matches = [
            doc for doc in final_documents if rule_id_to_search in doc.page_content
        ]
        matches.extend(rule_matches[:num_chunks])

    # Check for exact match search based on numbers
    if numbers_in_query:
        number_to_search = str(numbers_in_query[0])  # Convert to string for comparison
        number_matches = [
            doc for doc in final_documents if number_to_search in doc.page_content
        ]
        matches.extend(number_matches[:num_chunks])

    # Fall back to embedding-based search for more complex queries
    retriever = vector_store.as_retriever(search_kwargs={"k": num_chunks})
    similar_docs = retriever.get_relevant_documents(query)

    # Ensure embedding-based matches are added properly
    embedding_matches = [doc for doc in similar_docs if hasattr(doc, "page_content")]
    matches.extend(embedding_matches)

    # Return the matches
    return matches



# Define a function to run the LLM query pipeline
def run_llm(query: str, chat_history):
    """
    Main pipeline for processing user queries:
    1. Sets up vector store and LLM (Groq)
    2. Defines conversation prompts
    3. Performs rule-based search for exact matches
    4. Creates a retrieval chain that:
       - Considers chat history
       - Combines relevant documents
       - Generates a response using the LLM
    """
    # Initialize the embedding model to vectorize text.
    # The OpenAIEmbeddings class is used to generate embeddings for both the query and documents.
    # The `model="text-embedding-3-small"` specifies the particular embedding model to use.

    # Initialize the Pinecone vector store.
    # This creates a connection to the Pinecone index where pre-embedded documents are stored.
    # The `index_name` refers to the specific Pinecone index to be used.
    # The `embedding` parameter provides the model to match document embeddings with query embeddings.
    docsearch = Pinecone(index_name=INDEX_NAME, embedding=embeddings)

    # Set up the LLM for conversational responses.
    # `ChatOpenAI` initializes a chat-based language model with the specified parameters.
    # `verbose=True` ensures that additional processing details are logged for debugging.
    # `temperature=0` controls the randomness of the responses; a lower value makes outputs more deterministic.
    chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

    # Define the main conversation prompt template
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_template( 
    """
    I am Onase Helper Bot, your dedicated assistant for all application-related queries and healthcare information needs. I provide accurate, concise information while maintaining context awareness throughout our conversations.


    Instructions:
    Do not make up answers, if the answer is not in the given context, say I do not know.
    1. Answer questions in plain English and ensure your response is easy to understand for a doctor, always answer in bullet points if answer is more than few sentences.
    2. Always respond with according to my knowledge base, and so on...
    3. Always be consistent, if user asked same question as before, give him the same reply please.
    4. If user asks for password and username, do not share, say "I am not allowed to share this information"
    5. When asked to summarize, base the summary only on the relevant details from the conversation history. Ignore any newly retrieved chunks or external context for summarization tasks.
    6. For requests like "summarize the above information," focus only on the most recent exchanges in the conversation history. Extract and condense the key points into a concise response.
    7. When answering non-summarization queries, you may use the retrieved context along with the conversation history to provide accurate and complete responses.
    8. Use the retrieved documents to answer the user's query. If specific codes (e.g., BV-XXXXX) or numbers are included, ensure they are explicitly addressed and highlighted in the response.

    Conversation History:
    {context}

    Current Query:
    {input}
    """
)
    
    
        # Define prompt for rephrasing follow-up questions
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
 
    # Create retriever that's aware of conversation history
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    # Perform rule-based search and format results
    result = rule_based_search(query, docsearch, num_chunks=10)
    additional_context = "\n".join([doc.page_content for doc in result])
    
    # Create chain to combine documents into a response
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    
    # Combine query with any exact matches found
    query_with_context = f"{query}\n\nAdditional Context:\n{additional_context}"
    
    # Create and execute the final retrieval chain
    qa = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain)
    result = qa.invoke({
        "input": query_with_context,
        "chat_history": chat_history,
    })

    # Print the result for debugging purposes.
    # print(result)

    # Return the result to the caller.
    return result
