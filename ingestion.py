from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
import json
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Create an instance of the Pinecone class
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if the index exists, and create it if it doesn't
index_name = "rcm-final-app"  # Your index name
# Create an instance of the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Specify the model you want to use

# Now create the PineconeVectorStore with the embedding model
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)  # Pass the embedding model

def load_json_documents(folder_path):
    """
    Load and process JSON files into a list of Document objects.

    Args:
        file_paths (list of str): List of JSON file paths to process.

    Returns:
        list of Document: A list of Document objects with content and metadata.
    """
    combined_json_documents = []

    for file_path in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_path)
        try:
            # Read JSON file
            data = pd.read_json(file_path)
            
            # Convert each row in the JSON to a Document
            for _, row in data.iterrows():
                combined_json_documents.append(
                    Document(
                        page_content=str(row.to_dict()),
                        metadata={"file_type": "json", "source": file_path.split('/')[-1]}  # Add source dynamically
                    )
                )
        except ValueError as e:
            print(f"Error processing JSON file {file_path}: {e}")

    print(f"Loaded {len(combined_json_documents)} documents from JSON files.")
    return combined_json_documents

def load_pdf_documents(folder_path, domain):
    """
    Load and process all PDF files from the specified folder.

    This function iterates through all PDF files in the provided folder, extracts their content
    using the PyPDFLoader, and updates each document with relevant metadata such as file type
    and source file name.

    Args:
        folder_path (str): The path to the folder containing PDF files.

    Returns:
        list[Document]: A list of Document objects containing the content and metadata of all
        processed PDF files.

    Example:
        folder_path = "./PDF_Docs"
        documents = load_pdf_documents(folder_path)
        print(f"Total documents processed: {len(documents)}")
    """
    # Initialize a list to store all documents
    all_pdf_documents = []

    # Ensure the folder path exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder path {folder_path} does not exist.")

    # Iterate through each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):  # Process only PDF files
            file_path = os.path.join(folder_path, file_name)
            
            # Load PDF file
            pdf_loader = PyPDFLoader(file_path)
            pdf_documents = pdf_loader.load()
            
           # Add metadata to each document
            for doc in pdf_documents:
                doc.metadata.update({
                    "file_type": "pdf",
                    "source": file_name,
                    "domain": domain  # Add domain metadata
                })
            
            # Add documents to the main list
            all_pdf_documents.extend(pdf_documents)
            print(f"Loaded {len(pdf_documents)} documents from {file_name}.")

    print(f"Total {len(all_pdf_documents)} documents loaded from all PDFs.")

    # Return or process the combined documents as needed
    return all_pdf_documents


def ingest_docs():
    # Load PDF files
    rcm_pdf_folder_path = "./PDF_Docs/RCM Docs"
    dhis_pdf_folder_path = "./PDF_Docs/DHIS Docs"

    rcm_pdf_documents = load_pdf_documents(rcm_pdf_folder_path, domain="RCM")
    dhis_pdf_documents = load_pdf_documents(dhis_pdf_folder_path, domain="DHIS")

    pdf_documents = rcm_pdf_documents + dhis_pdf_documents
    print(f"Combined PDF documents from RCM and DHIS: {len(pdf_documents)} documents loaded.")

    # Load JSON Files
    json_folder_path = "./JSON_Documents"
    json_documents = load_json_documents(json_folder_path)

    # Combine all documents
    all_documents = pdf_documents + json_documents

    # Split documents into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

    split_documents = []
    for doc in all_documents:
        split_documents.extend(text_splitter.split_documents([doc]))

    print(f"Going to add {len(split_documents)} to Pinecone")
    for doc in split_documents:
        print(f"Document Metadata: {doc.metadata}, Content Length: {len(doc.page_content)}")

    # Index documents in Pinecone using PineconeVectorStore
    PineconeVectorStore.from_documents(
        split_documents, embeddings, index_name="rcm-final-app"
    )  # Updated indexing method

    print("****Loading to vectorstore done ***")

if __name__ == "__main__":
    ingest_docs()
