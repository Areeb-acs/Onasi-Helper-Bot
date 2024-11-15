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
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


INDEX_NAME = "rcm-final-app"
embeddings = OpenAIEmbeddings()

groq_api_key = os.getenv("GROQ_API_KEY")
# Initialize Pinecone
docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

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


def rule_based_search(query, vector_store, num_chunks=10):
    """
    Perform rule-based search on all documents retrieved from the vector store.
    """
    # Extract numbers or rule ID from the query
    numbers_in_query = extract_numbers_from_query(query)
    rule_id_match = re.search(r"BV-\d{5}", query)  # Adjust regex for your Rule ID format

    # Retrieve all documents from the vector store
    final_documents = get_all_documents(vector_store)

    # Check for Rule ID match
    if rule_id_match:
        rule_id_to_search = rule_id_match.group(0)  # Extract matched Rule ID
        matches = [
            doc
            for doc in final_documents
            if rule_id_to_search in doc.page_content  # Check if the Rule ID exists in the page content
        ]
        if matches:
            return matches[:num_chunks]  # Return top 'num_chunks' matches

    # Check for exact match search based on numbers
    if numbers_in_query:
        number_to_search = str(numbers_in_query[0])  # Convert to string for comparison
        matches = [
            doc
            for doc in final_documents
            if number_to_search in doc.page_content  # Check if the number exists in the page content
        ]
        if matches:
            return matches[:num_chunks]  # Return top 'num_chunks' matches

    # If no matches found, return an empty list
    return []


# Define a function to run the LLM query pipeline
def run_llm(query: str, chat_history):
    # Initialize the embedding model to vectorize text.
    # The OpenAIEmbeddings class is used to generate embeddings for both the query and documents.
    # The `model="text-embedding-3-small"` specifies the particular embedding model to use.

    # Initialize the Pinecone vector store.
    # This creates a connection to the Pinecone index where pre-embedded documents are stored.
    # The `index_name` refers to the specific Pinecone index to be used.
    # The `embedding` parameter provides the model to match document embeddings with query embeddings.
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # Set up the LLM for conversational responses.
    # `ChatOpenAI` initializes a chat-based language model with the specified parameters.
    # `verbose=True` ensures that additional processing details are logged for debugging.
    # `temperature=0` controls the randomness of the responses; a lower value makes outputs more deterministic.
    chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

    # Retrieve a prebuilt prompt template for retrieval-based question answering (QA).
    # This is done using `hub.pull`, which fetches a predefined prompt template from LangChain's repository.
    # retrieval_qa_chat_promot = hub.pull("langchain-ai/retrieval-qa-chat")
    # rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_template( 
    """
    You are a friendly conversational chatbot that remembers context across a conversation. Use the provided conversation history and context only to understand the user's question and provide clear, concise, and accurate responses for doctors.

    Instructions:
    1. Answer questions in plain English and ensure your response is easy to understand for a doctor.
    2. Always respond with according to my knowledge base, and so on...
    2. Always be consistent, if user asked same question as before, give him the same reply please.
    2. If user asks for password and username, do not share, say "I am not allowed to share this information"
    3. When asked to summarize, base the summary only on the relevant details from the conversation history. Ignore any newly retrieved chunks or external context for summarization tasks.
    4. For requests like "summarize the above information," focus only on the most recent exchanges in the conversation history. Extract and condense the key points into a concise response.
    5. When answering non-summarization queries, you may use the retrieved context along with the conversation history to provide accurate and complete responses.
    6. Use the retrieved documents to answer the user's query. If specific codes (e.g., BV-XXXXX) or numbers are included, ensure they are explicitly addressed and highlighted in the response.
    
    
    Conversation History:
    {context}

    Current Question:
    {input}
    
   
    """
)
    
    
        # rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    rephrase_prompt = ChatPromptTemplate.from_template( 
    """
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.


    Chat History:

    {chat_history}

    Follow Up Input: {input}

    Standalone Question:
    """
    )
 
    #
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    # Create a document combination chain using the retrieved prompt and LLM.
    # `create_stuff_documents_chain` combines multiple document results into a single cohesive response.
    # The `chat` is the LLM used for response generation, and the `retrieval_qa_chat_promot` is the template guiding how responses are structured.
    
    result = rule_based_search(query, docsearch, num_chunks=5)
        # Format the retrieved chunks as additional context
    additional_context = "\n".join([doc.page_content for doc in result])
    
    
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    # Append the additional context to the query
    query_with_context = f"{query}\n\nAdditional Context:\n{additional_context}"
    # Build a retrieval-based QA chain.
    # This chain first retrieves relevant documents from the Pinecone vector store using the `docsearch` retriever.
    # Then it uses the `stuff_documents_chain` to combine these documents into a single, coherent answer.
    qa = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain)
    
    # Invoke the QA chain
    result = qa.invoke({
        "input": query_with_context,
        "chat_history": chat_history,
    })


    # Print the result for debugging purposes.
    # print(result)

    # Return the result to the caller.
    return result
