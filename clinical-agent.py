from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

from typing import List, Dict
from langchain_chroma import Chroma
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain

file_path = "./heart.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

all_splits = text_splitter.split_documents(docs)

print(len(all_splits))


# Load the environment variables from the .env file
load_dotenv()



embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])
vector_store = Chroma(
    collection_name="example_collection", 
    embedding_function=embeddings,
    persist_directory="./chroma_heart_db"
)

# Add documents to the vector store
vector_store.add_documents(documents=all_splits)


   






# print(results[0])
# # results = vector_store.similarity_search(
#    retriever(query)
# )

# print(results[0])




# @chain
# def retriever(query: str) -> List[Document]:
#     result = vector_store.retrieve_info(query, k=1)
#     print("retruver")
#     for doc in result:
#         return result[0] if result else "No relevant information found."

# def retrieve_info(query: str):
#     return {
#         "agent": "clinical-agent",
#         "query": query,
#         "response": retriever.invoke(query)
#     }
# retriever.batch(
#     [
#         "provide sites for heart attacks",
#     ],
# )


