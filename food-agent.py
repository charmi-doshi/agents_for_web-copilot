from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

file_path = "./food.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

all_splits = text_splitter.split_documents(docs)

print(len(all_splits))
import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()



embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
ids = vector_store.add_documents(documents=all_splits)

# results = vector_store.similarity_search(
#     "average price of mil , whole grains from 1993 to this year"
# )

# print(results[0])
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain


@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


retriever.batch(
    [
        "average price of milk , whole grains from 1993 to this year",
    ],
)