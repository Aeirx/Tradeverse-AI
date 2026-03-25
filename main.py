import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 1. Imports from your own files
from algo_engine import run_ensemble_model

# 2. Load the Secret Keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

print("🔌 Connecting to Pinecone Cloud...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("tradeverse-news")

print("🧠 Waking up the AI Language Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Initialize the Server
app = FastAPI(title="Tradeverse AI Brain")

# 4. Define the formats for incoming requests
class SearchQuery(BaseModel):
    text: str

class WeightConfig(BaseModel):
    sentiment: float
    ma: float
    rsi: float

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
    if not search_results['matches']:
        return {"error": "No matching news found in memory."}
        
    best_match = search_results['matches'][0]
    
    # D. Send the answer back as JSON!
    return {
        "query": query.text,
        "best_headline": best_match['metadata']['text'],
        "confidence_score": round(best_match['score'], 2)
    }

@app.post("/trade-signal")
def get_custom_trade_signal(weights: WeightConfig):
    print(f"📡 Received custom user weights: {weights.model_dump()}")
    
    # Pass the user's exact UI slider values directly into your math engine!
    decision = run_ensemble_model({
        "sentiment": weights.sentiment,
        "ma": weights.ma,
        "rsi": weights.rsi
    })
    
    return decision