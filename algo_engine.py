import yfinance as yf
import pandas as pd
import numpy as np
import re
from transformers import pipeline

print("📊 Booting up the Weighted Quant Ensemble Engine with FinBERT...")

# Global pipeline initialization (happens when the module is imported)
try:
    print("🧠 Loading FinBERT Sentiment Engine (This might take a second first time)...")
    nlp_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception as e:
    print(f"⚠️ Failed to load FinBERT: {e}")
    nlp_pipeline = None

def get_live_technicals(symbol):
    try:
        # Fetch 3 months of data to ensure we can calculate 50 MA and RSI properly
        ticker = yf.Ticker(symbol, session=None) # session=None avoids caching bugs
        df = ticker.history(period="3mo")
        if df.empty:
            return None, None, None
            
        close = df['Close']
        if len(close) < 50:
            return 0.0, 0.0, 0.0 # Default neutral if not enough data
            
        # 1. 50-day SMA Momentum
        sma_50 = close.rolling(window=50).mean().iloc[-1]
        current_price = close.iloc[-1]
        
        # Raw percentage difference from 50 SMA (scaled)
        ma_diff_pct = (current_price - sma_50) / sma_50
        ma_score = min(max(ma_diff_pct * 10, -1.0), 1.0) 
        
        # 2. 14-day RSI (Wilder's RSI)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # RSI score: (Oversold < 30 is Bullish, Overbought > 70 is Bearish)
        rsi_score = (50 - current_rsi) / 20.0
        rsi_score = min(max(rsi_score, -1.0), 1.0)
        
        # 3. Volatility (Risk Management) - 20 day standard dev % of price
        returns = close.pct_change()
        volatility = returns.tail(20).std() * np.sqrt(252) # Annualized Volatility
        
        return float(ma_score), float(rsi_score), float(volatility)
        
    except Exception as e:
        print(f"⚠️ YFinance Error for {symbol}: {e}")
        return 0.0, 0.0, 0.0

def apply_fake_news_filter(headline, score):
    # Check for rumor/fake verbiage
    pattern = re.compile(r'\b(rumor|allegedly|scam|unverified|fraud|claims|falsely|supposedly)\b', re.IGNORECASE)
    if pattern.search(headline):
        print(f"🚨 FAKE NEWS REGEX TRIPPED! Unverified phrasing detected in: '{headline}'")
        # Slash score dampening by 80% to protect capital
        return score * 0.2
    return score

def get_finbert_sentiment(headline):
    if not headline or nlp_pipeline is None:
        return 0.0
        
    res = nlp_pipeline(headline)[0]
    label = res['label']  # 'positive', 'negative', 'neutral'
    confidence = res['score']
    
    if label == 'positive':
        score = confidence
    elif label == 'negative':
        score = -confidence
    else:
        score = 0.0
        
    # Apply semantic regex filter for Fake News protection
    score = apply_fake_news_filter(headline, score)
    return score

def run_ensemble_model(symbol, weights, headline=""):
    print(f"\n⚖️ Gathering votes from all Trading Algorithms for {symbol}...")

    # 1. AI NLP FinBERT Sentiment
    sent_score = get_finbert_sentiment(headline)
    
    # 2. Mathematical Indicators
    ma_score, rsi_score, volatility = get_live_technicals(symbol)
    
    # Fallback sanity check
    if ma_score is None:
        ma_score, rsi_score, volatility = 0.0, 0.0, 0.0
        print("⚠️ No pricing data found. Using neutral indicators.")
        
    print(f"📰 FinBERT AI Sentiment:    {sent_score:+.2f} (Headline: {headline[:40]}... )")
    print(f"📈 50 MA Momentum Algo:     {ma_score:+.2f}")
    print(f"📊 14-Day RSI Oscillation:  {rsi_score:+.2f}")
    print(f"📉 Annualized Volatility:   {volatility:.2%}")

    # 3. Risk Constraint Hard-Aborts
    if volatility > 0.80: # Exceeds 80% annualized volatility -> MEME STOCK ALERT
        print("\n⛔ [ABORT] EXTREME VOLATILITY DETECTED. RISK LIMITS EXCEEDED.")
        return { "final_score": 0.0, "signal": "⚪ HOLD" }
        
    # 4. Master Engine Weighting
    final_weighted_score = (
        (sent_score * weights['sentiment']) +
        (ma_score * weights['ma']) +
        (rsi_score * weights['rsi'])
    )

    # 5. Strict Confluence Check
    if final_weighted_score >= 0.20:
        if sent_score < 0:
             print("⚠️ [CAUTION] Positive indicators but negative news sentiment! Downgrading to HOLD.")
             signal = "⚪ HOLD"
        else:
             signal = "🟢 BUY"
    elif final_weighted_score <= -0.20:
        signal = "🔴 SELL"
    else:
        signal = "⚪ HOLD"

    print("\n==================================================")
    print(f"🎯 USER WEIGHTS: {weights['sentiment']*100}% News | {weights['ma']*100}% MA | {weights['rsi']*100}% RSI")
    print(f"🧠 FINAL WEIGHTED SCORE: {final_weighted_score:+.2f}")
    print(f"🚨 MASTER TRADING SIGNAL: {signal}")
    print("==================================================\n")

    return {
        "final_score": round(final_weighted_score, 2),
        "signal": signal
    }

# --- LOCAL FILE TEST ---
if __name__ == "__main__":
    user_weights = {
        "sentiment": 0.40,
        "ma": 0.40,
        "rsi": 0.20
    }
    # Test a fake news rumor about TSLA
    run_ensemble_model("TSLA", user_weights, headline="Rumor: Elon Musk allegedly steps down, market panics!")
    # Test a legitimate positive news about AAPL
    run_ensemble_model("AAPL", user_weights, headline="Apple smashes earnings expectations natively!")