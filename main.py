from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from typing import List
import openai
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os

from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI()

origins = [
    "http://localhost:3000",  # React development server URL
    "https://yourfrontenddomain.com",  # Add any other frontend URL that needs access
     "*"  # This allows all origins (be careful using this in production)
]

# Add the CORSMiddleware to your FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

load_dotenv()
# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Load OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Chroma Vector Store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# API Request Model
class QueryRequest(BaseModel):
    prompt: str
    model: str = "gpt-4"  # Default model

def is_heart_related(prompt: str):
    """Check if the prompt is related to heart diseases or conditions."""
    heart_keywords = ["heart", "cardiac", "cardiovascular", "coronary", "arrhythmia", "hypertension", "stroke"]
    return any(word in prompt.lower() for word in heart_keywords)

def retrieve_medical_info(query: str) -> str:
    """Retrieve relevant clinical information using ChromaDB."""
    result = vector_store.similarity_search(query, k=1)
    return result[0].page_content if result else "No relevant clinical information found."

def is_food_related(prompt: str):
    """Check if the prompt is related to heart diseases or conditions."""
    food_keywords = ["food", "milk", "vitamins", "food security", "nutrition"]
    return any(word in prompt.lower() for word in food_keywords)

def retrieve_food_info(query: str) -> str:
    """Retrieve relevant clinical information using ChromaDB."""
    result = vector_store.similarity_search(query, k=1)
    return result[0].page_content if result else "No relevant clinical information found."

@app.post("/query/")
async def query_llm(request: QueryRequest):
    try:
        if is_heart_related(request.prompt):
            # Fetch relevant clinical information from ChromaDB
            response = retrieve_medical_info(request.prompt)
            # response = subprocess.run(['python', 'heart.py', request.prompt],capture_output=True, text=True)
            return {"response": response}

        elif is_food_related(request.prompt):
            # Fetch relevant food-related information from ChromaDB
            response = retrieve_food_info(request.prompt)
        else:
            # Use OpenAI's GPT for general queries
            ai_response = client.chat.completions.create(
                model=request.model,
                messages=[{"role": "user", "content": request.prompt}]
            )
            response = ai_response.choices[0].message.content
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server: uvicorn main:app --reload
