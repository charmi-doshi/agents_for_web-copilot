# clinical_agent.py
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# Load the environment variables
load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# File path to the heart-related PDF
file_path = "./heart.pdf"

def load_and_process_pdf(file_path: str):
    """Loads and processes the PDF, splitting into chunks."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Text splitting into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)
    
    return all_splits

def retrieve_medical_info(query: str, all_splits) -> str:
    """Retrieve relevant clinical information using ChromaDB."""
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Initialize Chroma vector store
    vector_store = Chroma(
        collection_name="heart_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
    
    # Add documents to the vector store
    vector_store.add_documents(documents=all_splits)

    # Generate vector for the query
    query_vector = embeddings.embed_query(query)
    
    # Perform similarity search using the vector store
    result = vector_store.similarity_search(query_vector, k=1)  # Get the top 1 most similar document

    return result[0].page_content if result else "No relevant clinical information found."

def main(query: str):
    """Main function to process query."""
    all_splits = load_and_process_pdf(file_path)
    response = retrieve_medical_info(query, all_splits)
    return response

if __name__ == "__main__":
    query = sys.argv[1]  # Get the query from the command-line argument
    response = main(query)
    print(response)
