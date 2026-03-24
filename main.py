import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 1. Load the Secret Key
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

print("🔌 Connecting to Pinecone Cloud...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("tradeverse-news")

print("🧠 Waking up the AI Language Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Initialize the Server
app = FastAPI(title="Tradeverse AI Brain")

# 3. Define the format for incoming requests (Expect a JSON with a 'text' field)
class SearchQuery(BaseModel):
    text: str

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "online", "message": "🧠 AI Brain is listening for signals!"}

@app.post("/search")
def search_news(query: SearchQuery):
    print(f"📡 Received search request for: '{query.text}'")
    
    # A. Translate the incoming English text to Math
    query_vector = model.encode(query.text).tolist()
    
    # B. Search the Pinecone Cloud
    search_results = index.query(
        vector=query_vector,
        top_k=1,
        include_metadata=True
    )
    
    # C. Extract the best match
    best_match = search_results['matches'][0]
    
    # D. Send the answer back as JSON!
    return {
        "query": query.text,
        "best_headline": best_match['metadata']['text'],
        "confidence_score": round(best_match['score'], 2)
    }