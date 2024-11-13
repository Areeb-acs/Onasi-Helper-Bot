from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
import json
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import pandas as pd
import os

embeddings = OpenAIEmbeddings()



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

def load_pdf_documents(folder_path):
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
                doc.metadata.update({"file_type": "pdf", "source": file_name})
            
            # Add documents to the main list
            all_pdf_documents.extend(pdf_documents)
            print(f"Loaded {len(pdf_documents)} documents from {file_name}.")

    print(f"Total {len(all_pdf_documents)} documents loaded from all PDFs.")

    # Return or process the combined documents as needed
    return all_pdf_documents

def ingest_docs():
    # Load PDF file
    folder_path = "./PDF_Docs"  # Path to the folder containing Word docs
    pdf_documents = load_pdf_documents(folder_path)
    
    
    for doc in pdf_documents[:5]:
        print(f"Metadata: {doc.metadata}, Content: {doc.page_content[:100]}")

    
    # Load JSON Files
    folder_path = "./JSON_Documents"
    json_documents = load_json_documents(folder_path)

    # Output metadata and content for verification
    for doc in json_documents[:5]:  # Display first 5 documents
        print(f"Metadata: {doc.metadata}, Content: {doc.page_content[:100]}")
    
    # Combine all documents
    all_documents = pdf_documents + json_documents 

    # Split documents into chunks for embedding, using specified chunk size and overlap
    text_splitter  = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

    split_documents = []
    for doc in all_documents:
        split_documents.extend(text_splitter.split_documents([doc]))

    print(f"Going to add {len(split_documents)} to Pinecone")
    for doc in split_documents:
        print(f"Document Metadata: {doc.metadata}, Content Length: {len(doc.page_content)}")

    # Index documents in Pinecone
    PineconeVectorStore.from_documents(
        split_documents, embeddings, index_name="rcm-final-app"
    )
    print("****Loading to vectorstore done ***")

if __name__ == "__main__":
    ingest_docs()
