import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from fastapi.middleware.cors import CORSMiddleware
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

# --- ADD THIS CORS BLOCK ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Explicitly trust your React frontend!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Define the formats for incoming requests
class SearchQuery(BaseModel):
    text: str

class WeightConfig(BaseModel):
    sentiment: float
    ma: float
    rsi: float
    
class TradeRequest(BaseModel):
    symbol: str
    weights: WeightConfig

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

@app.post("/api/predict")
def predict_trade_signal(request: TradeRequest): 
    print(f"\n🚀 AI Engine activated for {request.symbol}!")
    
    # --- 1. SEARCH PINECONE FOR REAL NEWS ---
    query_text = f"financial news and market updates for {request.symbol}"
    query_vector = model.encode(query_text).tolist()
    
    search_results = index.query(
        vector=query_vector,
        top_k=1,
        include_metadata=True
    )
    
    if search_results['matches']:
        best_match = search_results['matches'][0]
        headline = best_match['metadata']['text']
        # Pinecone cosine scores are usually 0 to 1. We shift it to -1 to +1 for our math engine.
        live_news_score = (best_match['score'] * 2) - 1.0 
        print(f"📡 PINECONE MEMORY: Found headline -> '{headline}'")
    else:
        print("📡 PINECONE MEMORY: No news found for this ticker.")
        live_news_score = 0.0 # Neutral if we have no news
        
    # --- 2. RUN THE QUANTITATIVE ENGINE ---
    decision_data = run_ensemble_model(
        weights={
            "sentiment": request.weights.sentiment,
            "ma": request.weights.ma,
            "rsi": request.weights.rsi
        },
        live_news_score=live_news_score
    )
    
    raw_signal = decision_data["signal"] 
    confidence = min(round(abs(decision_data["final_score"]) * 100, 1) + 60.0, 99.9)

    return {
        "signal": raw_signal,
        "confidence": confidence,
        "symbol": request.symbol
    }