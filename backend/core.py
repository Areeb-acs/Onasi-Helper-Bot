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


def retrieve_relevant_chunks(question, docsearch, num_chunks=10, file_type=None):
    import re

    # Extract numbers and Rule IDs from the query
    numbers_in_query = re.findall(r'\d+', question)  # Extract all numbers
    rule_id_match = re.search(r'\bBV-\d+\b', question)  # Match Rule IDs like "BV-00027"

    matches = []  # Initialize an empty list to hold all matches

    # Check for exact match search based on Rule ID
    if rule_id_match:
        rule_id_to_search = rule_id_match.group(0)  # Extract the matched Rule ID
        rule_id_matches = docsearch.similarity_search(rule_id_to_search, k=num_chunks)
   
        matches.extend(rule_id_matches)  # Add Rule ID matches to the results list

    # Check for exact match search based on numbers
    if numbers_in_query:
        for number in numbers_in_query:  # Iterate through all numbers found
            number_matches = docsearch.similarity_search(number, k=num_chunks)
            
            # Filter results to ensure the exact number is present in the content
            exact_matches = [
                match for match in number_matches if str(number) in match.page_content
            ]
            
            # Debugging: Print the exact matches for verification
            print(f"Exact matches for number {number}: {exact_matches}")

            # Add the exact matches to the results list
            matches.extend(exact_matches)


    # Concatenate all retrieved document contents into a single string
    return matches


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
    chat = ChatOpenAI(verbose=True, temperature=0)

    # Retrieve a prebuilt prompt template for retrieval-based question answering (QA).
    # This is done using `hub.pull`, which fetches a predefined prompt template from LangChain's repository.
    # retrieval_qa_chat_promot = hub.pull("langchain-ai/retrieval-qa-chat")
    # rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_template( 
    """
    You are a friendly conversational chatbot that remembers context across a conversation. Use the provided conversation history and context only to understand the user's question and provide clear, concise, and accurate responses for doctors.

    Instructions:
    1. Always refer to the conversation history for context and maintain continuity in your responses, only answer based on given context.
    2. Answer questions in plain English and ensure your response is easy to understand for a doctor.
    3. If user asks for password and username, do not share, say "I am not allowed to share this information"
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
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    # Build a retrieval-based QA chain.
    # This chain first retrieves relevant documents from the Pinecone vector store using the `docsearch` retriever.
    # Then it uses the `stuff_documents_chain` to combine these documents into a single, coherent answer.
    qa = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain)
    # Process query
    # Run the query through the QA chain.
    # The `invoke` method takes an input query and processes it through the chain.
    # The result contains the final response generated by the model.
    # Retrieve relevant chunks using exact matching
    # similar_matches = retrieve_relevant_chunks(query, docsearch, num_chunks=10)

    # # Process `similar_matches` to extract text content
    # similar_matches_content = [match.page_content for match in similar_matches]
    # Append the processed `similar_matches_content` to the `chat_history`
    # Append the retrieved content to the query
    # if similar_matches_content:
    #     query += "\nAdditional Relevant Chunks:\n" + "\n".join(similar_matches_content)
    # Invoke the QA chain with custom context if results exist
    # Ensure similar_matches is appended to chat_history properly
        # Retrieve relevant documents for the query

    
    # Invoke the QA chain
    result = qa.invoke({
        "input": query,
        "chat_history": chat_history,
    })


    # Print the result for debugging purposes.
    # print(result)

    # Return the result to the caller.
    return result
