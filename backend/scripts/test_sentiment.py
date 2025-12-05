"""
Test NLP Sentiment Factor

Tests news sentiment analysis using FinBERT or keyword fallback.
"""
import asyncio
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def test_finbert_availability():
    """Check if FinBERT is available."""
    print("\n=== Checking FinBERT Availability ===")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        print("✅ transformers + torch installed")
        print(f"   PyTorch version: {torch.__version__}")
        return True
    except ImportError as e:
        print(f"⚠️ FinBERT not available: {e}")
        print("   Using keyword-based fallback")
        return False


def test_keyword_sentiment():
    """Test keyword-based sentiment fallback."""
    print("\n=== Testing Keyword Sentiment ===")
    
    from quant.features.sentiment import FinBERTSentiment
    
    analyzer = FinBERTSentiment()
    
    test_headlines = [
        "Apple stock surges after record-breaking earnings beat",
        "Tesla shares plunge on disappointing delivery numbers",
        "Microsoft announces quarterly dividend",
        "NVIDIA rallies on AI chip demand, beating estimates by 10%",
        "Amazon warns of slowing growth amid economic concerns"
    ]
    
    for headline in test_headlines:
        scores = analyzer._keyword_fallback(headline)
        net = scores['positive'] - scores['negative']
        sentiment = "🟢" if net > 0.1 else "🔴" if net < -0.1 else "🟡"
        print(f"{sentiment} Net: {net:+.2f} | {headline[:50]}...")
    
    return True


def test_news_sentiment():
    """Test full news sentiment factor."""
    print("\n=== Testing News Sentiment Factor ===")
    
    from quant.features.sentiment import NewsSentimentFactor
    
    factor = NewsSentimentFactor(lookback_days=7, max_articles=5)
    
    tickers = ["AAPL", "NVDA", "TSLA", "MSFT", "META"]
    
    results = []
    
    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        
        # Get score
        score_series = factor.compute(pd.DataFrame(), None, ticker=ticker)
        score = score_series.iloc[-1]
        
        results.append({'ticker': ticker, 'sentiment': score})
        
        # Get details
        details = factor.get_news_details(ticker)
        if details:
            print(f"Latest headlines:")
            for i, d in enumerate(details[:3]):
                sentiment = d['sentiment']
                icon = "🟢" if sentiment['net'] > 0.1 else "🔴" if sentiment['net'] < -0.1 else "🟡"
                print(f"  {icon} {d['headline'][:60]}...")
        
        icon = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "🟡"
        print(f"Aggregate Score: {score:+.3f} {icon}")
    
    # Summary
    print("\n=== Sentiment Summary ===")
    df = pd.DataFrame(results).sort_values('sentiment', ascending=False)
    for _, row in df.iterrows():
        icon = "🟢" if row['sentiment'] > 0.1 else "🔴" if row['sentiment'] < -0.1 else "🟡"
        print(f"{row['ticker']:6s}: {row['sentiment']:+.3f} {icon}")
    
    return True


def main():
    print("=" * 60)
    print("NLP SENTIMENT FACTOR TEST")
    print("=" * 60)
    
    finbert_available = test_finbert_availability()
    test_keyword_sentiment()
    test_news_sentiment()
    
    print("\n" + "=" * 60)
    if finbert_available:
        print("✅ FINBERT MODE ACTIVE")
    else:
        print("⚠️ KEYWORD FALLBACK MODE (install transformers for FinBERT)")
    print("✅ SENTIMENT FACTOR TEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
