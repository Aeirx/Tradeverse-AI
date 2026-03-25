from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("📊 Booting up the Weighted Quant Ensemble Engine...")

# --- THE 3 INDEPENDENT ALGORITHMS ---

def get_sentiment_score():
    # Imagine this is your Vader AI reading Finnhub news
    # It reads great news today, so it votes highly positive.
    return 0.80  

def get_moving_average_score():
    # Imagine this reads your market_data.csv
    # It sees the stock price is dropping below the 50-day average. Bearish.
    return -0.50 

def get_rsi_score():
    # RSI (Relative Strength Index) checks if a stock is overbought
    # It sees the stock is a little overbought. Slightly bearish.
    return -0.10 


# --- THE MASTER DECISION ENGINE ---

def run_ensemble_model(weights):
    print("\n⚖️ Gathering votes from all Trading Algorithms...")

    # 1. Get raw scores from the algos
    sent_score = get_sentiment_score()
    ma_score = get_moving_average_score()
    rsi_score = get_rsi_score()

    print(f"📰 News Sentiment Algo:  {sent_score:+.2f}")
    print(f"📈 Moving Average Algo:  {ma_score:+.2f}")
    print(f"📊 RSI Momentum Algo:    {rsi_score:+.2f}")

    # 2. Apply the user's custom weights!
    # (Score * Weight)
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
    # Test Case 1: The user trusts the Technicals (Moving Averages) over the News
    # 30% Sentiment, 60% MA, 10% RSI
    user_weights = {
        "sentiment": 0.30,
        "ma": 0.60,
        "rsi": 0.10
    }
    
    run_ensemble_model(user_weights)