print("📊 Booting up the Weighted Quant Ensemble Engine...")

# --- THE TECHNICAL ALGORITHMS ---

def get_moving_average_score():
    # Imagine this reads your market_data.csv
    # It sees the stock price is dropping below the 50-day average. Bearish.
    return -0.50 

def get_rsi_score():
    # RSI (Relative Strength Index) checks if a stock is overbought
    # It sees the stock is a little overbought. Slightly bearish.
    return -0.10 


# --- THE MASTER DECISION ENGINE ---
# Notice it now requires 'live_news_score' from Pinecone!
def run_ensemble_model(weights, live_news_score):
    print("\n⚖️ Gathering votes from all Trading Algorithms...")

    # 1. Get raw scores from the algos
    sent_score = live_news_score  # <-- This is now REAL AI data!
    ma_score = get_moving_average_score()
    rsi_score = get_rsi_score()

    print(f"📰 Real AI News Sentiment:  {sent_score:+.2f}")
    print(f"📈 Moving Average Algo:     {ma_score:+.2f}")
    print(f"📊 RSI Momentum Algo:       {rsi_score:+.2f}")

    # 2. Apply the user's custom weights!
    final_weighted_score = (
        (sent_score * weights['sentiment']) +
        (ma_score * weights['ma']) +
        (rsi_score * weights['rsi'])
    )

    # 3. The Master Decision Logic
    if final_weighted_score >= 0.20:
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

# --- TEST THE ENGINE ---
if __name__ == "__main__":
    user_weights = {
        "sentiment": 0.30,
        "ma": 0.60,
        "rsi": 0.10
    }
    # Passing a dummy live score of 0.80 just for the direct file test
    run_ensemble_model(user_weights, live_news_score=0.80)