"""
NLP Sentiment Factor using FinBERT

Uses HuggingFace's ProsusAI/finbert model to analyze financial news headlines.

FinBERT is a pre-trained NLP model specifically for financial text, providing
sentiment scores: positive, negative, neutral.

Fallback: If transformers not installed, uses keyword-based sentiment.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from quant.features.base import FeatureGenerator
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

# Try to import transformers for FinBERT
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    logger.warning("transformers not installed. Using keyword-based sentiment fallback.")
    logger.warning("Install with: pip install transformers torch")


class FinBERTSentiment:
    """
    FinBERT-based sentiment analyzer for financial text.
    
    Uses ProsusAI/finbert model from HuggingFace.
    Returns sentiment scores: positive, negative, neutral
    """
    
    _instance = None
    _model = None
    _tokenizer = None
    _is_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._is_loaded and FINBERT_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load FinBERT model (lazy loading for memory efficiency)."""
        if self._is_loaded:
            return
        
        try:
            logger.info("Loading FinBERT model (this may take a moment)...")
            self._tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self._model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self._model.eval()  # Set to evaluation mode
            self._is_loaded = True
            logger.info("FinBERT model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            self._is_loaded = False
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Returns:
            Dict with 'positive', 'negative', 'neutral' scores (sum to 1)
        """
        if not FINBERT_AVAILABLE or not self._is_loaded:
            return self._keyword_fallback(text)
        
        try:
            # Tokenize
            inputs = self._tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512, 
                padding=True
            )
            
            # Inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs = probs.numpy()[0]
            
            # FinBERT labels: positive, negative, neutral
            return {
                'positive': float(probs[0]),
                'negative': float(probs[1]),
                'neutral': float(probs[2])
            }
            
        except Exception as e:
            logger.debug(f"FinBERT inference failed: {e}")
            return self._keyword_fallback(text)
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze multiple texts efficiently."""
        return [self.analyze(text) for text in texts]
    
    def _keyword_fallback(self, text: str) -> Dict[str, float]:
        """Simple keyword-based sentiment when FinBERT unavailable."""
        text_lower = text.lower()
        
        positive_words = [
            'beat', 'beats', 'surge', 'surges', 'gain', 'gains', 'profit', 'profits',
            'growth', 'grow', 'rise', 'rises', 'rising', 'up', 'high', 'higher',
            'bullish', 'upgrade', 'upgraded', 'strong', 'outperform', 'buy',
            'record', 'breakthrough', 'innovation', 'success', 'positive',
            'boost', 'boosted', 'jump', 'jumps', 'soar', 'soars', 'rally',
            'optimistic', 'confidence', 'exceed', 'exceeds', 'exceeded'
        ]
        
        negative_words = [
            'miss', 'misses', 'fall', 'falls', 'drop', 'drops', 'loss', 'losses',
            'decline', 'declines', 'down', 'low', 'lower', 'weak', 'bearish',
            'downgrade', 'downgraded', 'sell', 'underperform', 'cut', 'cuts',
            'warning', 'concern', 'concerns', 'risk', 'risks', 'slump',
            'crash', 'plunge', 'plunges', 'tumble', 'tumbles', 'slide',
            'pessimistic', 'disappointing', 'disappoints', 'layoff', 'layoffs'
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count + 1  # +1 to avoid division by zero
        
        pos_score = pos_count / total
        neg_score = neg_count / total
        neutral_score = 1 - pos_score - neg_score
        
        return {
            'positive': pos_score,
            'negative': neg_score,
            'neutral': max(0, neutral_score)
        }


class NewsSentimentFactor(FeatureGenerator):
    """
    News Sentiment Factor using FinBERT.
    
    Aggregates sentiment from recent news headlines for a ticker.
    Higher score = more positive sentiment = expected outperformance.
    """
    
    def __init__(self, lookback_days: int = 7, max_articles: int = 10):
        """
        Args:
            lookback_days: How far back to fetch news
            max_articles: Maximum articles to analyze (for speed)
        """
        self.lookback_days = lookback_days
        self.max_articles = max_articles
        self.analyzer = FinBERTSentiment() if FINBERT_AVAILABLE else None
    
    @property
    def name(self) -> str:
        return "NewsSentimentFactor"
    
    @property
    def description(self) -> str:
        return "Aggregated news sentiment from FinBERT. Positive sentiment expected to outperform."
    
    def compute(
        self, 
        history: pd.DataFrame, 
        ticker_info: Optional[dict] = None,
        ticker: Optional[str] = None
    ) -> pd.Series:
        """
        Compute news sentiment score for a ticker.
        
        Returns:
            pd.Series with sentiment score (-1 to +1)
        """
        if ticker is None:
            return pd.Series([0.0], index=[pd.Timestamp.now()])
        
        try:
            # Fetch news from yfinance
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news or len(news) == 0:
                logger.debug(f"{ticker}: No news available")
                return pd.Series([0.0], index=[pd.Timestamp.now()])
            
            # Filter recent news and limit count
            cutoff = datetime.now() - timedelta(days=self.lookback_days)
            recent_news = []
            
            for article in news[:self.max_articles]:
                # Handle both old format and new nested format
                # New format: article['content']['title']
                # Old format: article['title']
                if isinstance(article, dict):
                    if 'content' in article:
                        content = article['content']
                        title = content.get('title', '')
                        pub_time = content.get('pubDate', '')
                        if pub_time:
                            try:
                                pub_date = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
                                if pub_date.replace(tzinfo=None) < cutoff:
                                    continue
                            except:
                                pass
                    else:
                        title = article.get('title', '')
                        pub_time = article.get('providerPublishTime', 0)
                        if pub_time:
                            pub_date = datetime.fromtimestamp(pub_time)
                            if pub_date < cutoff:
                                continue
                else:
                    continue
                
                if title:
                    recent_news.append(title)
            
            if not recent_news:
                logger.debug(f"{ticker}: No recent news found")
                return pd.Series([0.0], index=[pd.Timestamp.now()])
            
            # Analyze sentiment
            sentiments = []
            for headline in recent_news:
                if self.analyzer:
                    scores = self.analyzer.analyze(headline)
                else:
                    # Fallback
                    scores = FinBERTSentiment()._keyword_fallback(headline)
                
                # Net sentiment = positive - negative
                net = scores['positive'] - scores['negative']
                sentiments.append(net)
            
            # Aggregate: mean sentiment with recency weighting
            # More recent articles get higher weight (linear decay)
            weights = np.linspace(1.0, 0.5, len(sentiments))  # Newer = higher weight
            weighted_sentiment = np.average(sentiments, weights=weights)
            
            # Add article count factor (more articles = more confident)
            confidence = min(1.0, len(sentiments) / 5)  # Max confidence at 5+ articles
            final_score = weighted_sentiment * (0.5 + 0.5 * confidence)
            
            logger.debug(
                f"{ticker}: {len(recent_news)} articles, "
                f"Avg Sentiment={weighted_sentiment:.3f}, Final Score={final_score:.3f}"
            )
            
            return pd.Series([final_score], index=[pd.Timestamp.now()])
            
        except Exception as e:
            logger.warning(f"News sentiment failed for {ticker}: {e}")
            return pd.Series([0.0], index=[pd.Timestamp.now()])
    
    def get_news_details(self, ticker: str) -> List[Dict]:
        """
        Get detailed news analysis for a ticker.
        
        Returns list of dicts with headline, sentiment, publish time.
        """
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return []
            
            results = []
            for article in news[:self.max_articles]:
                # Handle both old and new nested format
                if isinstance(article, dict):
                    if 'content' in article:
                        content = article['content']
                        title = content.get('title', '')
                        publisher = content.get('provider', {}).get('displayName', 'Unknown')
                        pub_time = content.get('pubDate', '')
                        try:
                            pub_date = datetime.fromisoformat(pub_time.replace('Z', '+00:00')) if pub_time else None
                        except:
                            pub_date = None
                    else:
                        title = article.get('title', '')
                        publisher = article.get('publisher', 'Unknown')
                        pub_time = article.get('providerPublishTime', 0)
                        pub_date = datetime.fromtimestamp(pub_time) if pub_time else None
                else:
                    continue
                
                if not title:
                    continue
                
                if self.analyzer:
                    scores = self.analyzer.analyze(title)
                else:
                    scores = FinBERTSentiment()._keyword_fallback(title)
                
                results.append({
                    'headline': title,
                    'publisher': publisher,
                    'published': pub_date.isoformat() if pub_date else None,
                    'sentiment': {
                        'positive': round(scores['positive'], 3),
                        'negative': round(scores['negative'], 3),
                        'neutral': round(scores['neutral'], 3),
                        'net': round(scores['positive'] - scores['negative'], 3)
                    }
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get news details for {ticker}: {e}")
            return []


class EarningsCallSentiment(FeatureGenerator):
    """
    Sentiment from earnings call transcripts.
    
    Note: Requires access to transcript data (not freely available).
    This is a placeholder for future integration with paid data sources.
    """
    
    def __init__(self):
        self.analyzer = FinBERTSentiment() if FINBERT_AVAILABLE else None
    
    @property
    def name(self) -> str:
        return "EarningsCallSentiment"
    
    @property
    def description(self) -> str:
        return "Sentiment from earnings call transcripts (placeholder for paid data)."
    
    def compute(
        self, 
        history: pd.DataFrame, 
        ticker_info: Optional[dict] = None,
        transcript: Optional[str] = None
    ) -> pd.Series:
        """
        Compute sentiment from earnings call transcript.
        
        Args:
            transcript: Full earnings call transcript text
        """
        if not transcript:
            return pd.Series([0.0], index=[pd.Timestamp.now()])
        
        # Split transcript into chunks (FinBERT max 512 tokens)
        chunk_size = 400  # Approximate words per chunk
        words = transcript.split()
        chunks = [
            ' '.join(words[i:i+chunk_size]) 
            for i in range(0, len(words), chunk_size)
        ]
        
        if not chunks:
            return pd.Series([0.0], index=[pd.Timestamp.now()])
        
        # Analyze each chunk
        sentiments = []
        for chunk in chunks[:10]:  # Limit chunks analyzed
            if self.analyzer:
                scores = self.analyzer.analyze(chunk)
            else:
                scores = FinBERTSentiment()._keyword_fallback(chunk)
            
            net = scores['positive'] - scores['negative']
            sentiments.append(net)
        
        # Aggregate
        avg_sentiment = np.mean(sentiments)
        
        return pd.Series([avg_sentiment], index=[pd.Timestamp.now()])
