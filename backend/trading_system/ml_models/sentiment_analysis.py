"""
Advanced Sentiment Analysis System for Market Intelligence
Using free APIs (Reddit, Alpha Vantage) and local BERT models
"""

import numpy as np
import pandas as pd
import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
import os
import re
from collections import defaultdict, deque
import time

# NLP and Sentiment Analysis
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from flair.data import Sentence
from flair.models import TextClassifier
import yfinance as yf
from alpha_vantage.alpha_vantage import AlphaVantage
from newsapi import NewsApiClient

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Sentiment score data structure"""
    compound: float  # Overall sentiment (-1 to 1)
    positive: float  # Positive sentiment (0 to 1)
    negative: float  # Negative sentiment (0 to 1) 
    neutral: float   # Neutral sentiment (0 to 1)
    confidence: float  # Confidence in the analysis (0 to 1)
    source: str      # Source of the text (reddit, news, etc.)
    timestamp: datetime
    text_sample: str  # Sample of analyzed text


@dataclass
class MarketSentimentData:
    """Aggregated market sentiment data"""
    overall_sentiment: float
    confidence_level: float
    sentiment_trend: str  # bullish, bearish, neutral
    source_breakdown: Dict[str, float]
    volume_weighted_sentiment: float
    fear_greed_index: float
    social_sentiment: float
    news_sentiment: float
    timestamp: datetime
    sample_size: int


class MarketSentimentAnalyzer:
    """
    Comprehensive market sentiment analyzer using multiple free sources
    """
    
    def __init__(self, 
                 reddit_client_id: Optional[str] = None,
                 reddit_client_secret: Optional[str] = None,
                 alpha_vantage_key: Optional[str] = None,
                 news_api_key: Optional[str] = None):
        
        # Initialize sentiment analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize local BERT model for financial sentiment
        try:
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.finbert_pipeline = pipeline(
                "text-classification", 
                model=self.finbert_model, 
                tokenizer=self.finbert_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            self.finbert_available = True
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"FinBERT not available: {e}")
            self.finbert_available = False
        
        # Initialize Flair sentiment
        try:
            self.flair_classifier = TextClassifier.load('en-sentiment')
            self.flair_available = True
            logger.info("Flair sentiment classifier loaded")
        except Exception as e:
            logger.warning(f"Flair not available: {e}")
            self.flair_available = False
        
        # Initialize API clients with fallback support
        self.reddit_client = None
        self.alpha_vantage_client = None
        self.news_client = None
        
        # Reddit API
        if reddit_client_id and reddit_client_secret:
            try:
                self.reddit_client = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent="TradingBot:v1.0"
                )
                logger.info("Reddit API client initialized")
            except Exception as e:
                logger.warning(f"Reddit API initialization failed: {e}")
        
        # Alpha Vantage API
        if alpha_vantage_key:
            try:
                self.alpha_vantage_client = AlphaVantage(key=alpha_vantage_key, output_format='pandas')
                logger.info("Alpha Vantage API client initialized")
            except Exception as e:
                logger.warning(f"Alpha Vantage initialization failed: {e}")
        
        # News API
        if news_api_key:
            try:
                self.news_client = NewsApiClient(api_key=news_api_key)
                logger.info("News API client initialized")
            except Exception as e:
                logger.warning(f"News API initialization failed: {e}")
        
        # Caching for API rate limiting
        self.sentiment_cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Sentiment history
        self.sentiment_history = deque(maxlen=1000)
        
        logger.info("Market Sentiment Analyzer initialized")
    
    async def analyze_market_sentiment(self, symbols: List[str] = None) -> MarketSentimentData:
        """
        Comprehensive market sentiment analysis from multiple sources
        """
        try:
            symbols = symbols or ['BTC', 'ETH', 'cryptocurrency', 'bitcoin', 'ethereum']
            
            # Collect sentiment from all available sources
            sentiment_scores = []
            
            # Social media sentiment (Reddit)
            if self.reddit_client:
                try:
                    reddit_sentiment = await self._analyze_reddit_sentiment(symbols)
                    sentiment_scores.extend(reddit_sentiment)
                except Exception as e:
                    logger.warning(f"Reddit sentiment analysis failed: {e}")
            
            # News sentiment
            news_sentiment = await self._analyze_news_sentiment(symbols)
            sentiment_scores.extend(news_sentiment)
            
            # Fear & Greed Index (from public APIs)
            fear_greed_score = await self._get_fear_greed_index()
            
            # Market data sentiment (price action analysis)
            market_data_sentiment = await self._analyze_market_data_sentiment(symbols)
            sentiment_scores.extend(market_data_sentiment)
            
            # Aggregate all sentiment scores
            aggregated_sentiment = self._aggregate_sentiment_scores(sentiment_scores)
            
            # Create comprehensive market sentiment data
            market_sentiment = MarketSentimentData(
                overall_sentiment=aggregated_sentiment['overall'],
                confidence_level=aggregated_sentiment['confidence'],
                sentiment_trend=self._determine_sentiment_trend(aggregated_sentiment['overall']),
                source_breakdown=aggregated_sentiment['source_breakdown'],
                volume_weighted_sentiment=aggregated_sentiment.get('volume_weighted', 0.0),
                fear_greed_index=fear_greed_score,
                social_sentiment=aggregated_sentiment.get('social', 0.0),
                news_sentiment=aggregated_sentiment.get('news', 0.0),
                timestamp=datetime.utcnow(),
                sample_size=len(sentiment_scores)
            )
            
            # Store in history
            self.sentiment_history.append(market_sentiment)
            
            return market_sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            # Return neutral sentiment as fallback
            return MarketSentimentData(
                overall_sentiment=0.0,
                confidence_level=0.1,
                sentiment_trend='neutral',
                source_breakdown={},
                volume_weighted_sentiment=0.0,
                fear_greed_index=50.0,
                social_sentiment=0.0,
                news_sentiment=0.0,
                timestamp=datetime.utcnow(),
                sample_size=0
            )
    
    async def _analyze_reddit_sentiment(self, symbols: List[str]) -> List[SentimentScore]:
        """Analyze sentiment from Reddit cryptocurrency subreddits"""
        sentiment_scores = []
        
        try:
            # Define subreddits to monitor
            subreddits = ['cryptocurrency', 'Bitcoin', 'ethereum', 'CryptoMarkets', 'altcoin']
            
            # Search terms for each symbol
            search_terms = []
            for symbol in symbols:
                search_terms.extend([symbol.lower(), symbol.upper(), f"${symbol}"])
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    
                    # Get hot posts
                    hot_posts = list(subreddit.hot(limit=50))
                    
                    for post in hot_posts:
                        # Check if post is relevant to our symbols
                        post_text = f"{post.title} {post.selftext}".lower()
                        if any(term.lower() in post_text for term in search_terms):
                            
                            # Analyze post sentiment
                            post_sentiment = self._analyze_text_sentiment(
                                post.title + " " + post.selftext,
                                source=f"reddit_{subreddit_name}"
                            )
                            
                            # Weight by upvotes (engagement)
                            engagement_weight = min(post.ups / 100, 10)  # Cap at 10x weight
                            post_sentiment.confidence *= (1 + engagement_weight * 0.1)
                            
                            sentiment_scores.append(post_sentiment)
                            
                            # Analyze top comments
                            post.comments.replace_more(limit=0)
                            for comment in post.comments[:10]:  # Top 10 comments
                                if len(comment.body) > 20:  # Filter short comments
                                    comment_sentiment = self._analyze_text_sentiment(
                                        comment.body,
                                        source=f"reddit_{subreddit_name}_comment"
                                    )
                                    
                                    # Weight by comment score
                                    comment_weight = min(comment.score / 10, 5)
                                    comment_sentiment.confidence *= (1 + comment_weight * 0.05)
                                    
                                    sentiment_scores.append(comment_sentiment)
                
                except Exception as e:
                    logger.warning(f"Error analyzing subreddit {subreddit_name}: {e}")
                    continue
            
            logger.info(f"Analyzed {len(sentiment_scores)} Reddit posts/comments")
            
        except Exception as e:
            logger.error(f"Reddit sentiment analysis error: {e}")
        
        return sentiment_scores
    
    async def _analyze_news_sentiment(self, symbols: List[str]) -> List[SentimentScore]:
        """Analyze sentiment from financial news"""
        sentiment_scores = []
        
        try:
            # Yahoo Finance news (free)
            for symbol in symbols[:3]:  # Limit to avoid rate limits
                try:
                    # Get news from Yahoo Finance
                    if symbol.upper() in ['BTC', 'BITCOIN']:
                        ticker = yf.Ticker("BTC-USD")
                    elif symbol.upper() in ['ETH', 'ETHEREUM']:
                        ticker = yf.Ticker("ETH-USD")
                    else:
                        continue
                    
                    news = ticker.news
                    
                    for article in news[:10]:  # Analyze top 10 articles
                        title = article.get('title', '')
                        summary = article.get('summary', '')
                        
                        if title or summary:
                            article_text = f"{title} {summary}"
                            article_sentiment = self._analyze_text_sentiment(
                                article_text,
                                source="yahoo_finance_news"
                            )
                            sentiment_scores.append(article_sentiment)
                
                except Exception as e:
                    logger.warning(f"Error getting Yahoo Finance news for {symbol}: {e}")
                    continue
            
            # News API (if available)
            if self.news_client:
                try:
                    # Search for cryptocurrency news
                    news_articles = self.news_client.get_everything(
                        q='cryptocurrency OR bitcoin OR ethereum',
                        language='en',
                        sort_by='publishedAt',
                        from_param=(datetime.utcnow() - timedelta(days=1)).isoformat(),
                        page_size=50
                    )
                    
                    for article in news_articles.get('articles', [])[:20]:
                        title = article.get('title', '')
                        description = article.get('description', '')
                        
                        if title or description:
                            article_text = f"{title} {description}"
                            article_sentiment = self._analyze_text_sentiment(
                                article_text,
                                source="news_api"
                            )
                            sentiment_scores.append(article_sentiment)
                
                except Exception as e:
                    logger.warning(f"News API error: {e}")
            
            # Alpha Vantage news (if available)
            if self.alpha_vantage_client:
                try:
                    # Get market news
                    news_data, _ = self.alpha_vantage_client.get_news_sentiment()
                    
                    for article in news_data.iloc[:20].itertuples():  # Analyze top 20
                        if hasattr(article, 'title') and hasattr(article, 'summary'):
                            article_text = f"{article.title} {article.summary}"
                            article_sentiment = self._analyze_text_sentiment(
                                article_text,
                                source="alpha_vantage_news"
                            )
                            sentiment_scores.append(article_sentiment)
                
                except Exception as e:
                    logger.warning(f"Alpha Vantage news error: {e}")
            
            logger.info(f"Analyzed {len(sentiment_scores)} news articles")
            
        except Exception as e:
            logger.error(f"News sentiment analysis error: {e}")
        
        return sentiment_scores
    
    async def _get_fear_greed_index(self) -> float:
        """Get Fear & Greed Index from free API"""
        try:
            # Use free Fear & Greed Index API
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.alternative.me/fng/') as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and len(data['data']) > 0:
                            return float(data['data'][0]['value'])
            
            # Fallback to neutral value
            return 50.0
            
        except Exception as e:
            logger.warning(f"Error getting Fear & Greed Index: {e}")
            return 50.0
    
    async def _analyze_market_data_sentiment(self, symbols: List[str]) -> List[SentimentScore]:
        """Analyze sentiment based on market data patterns"""
        sentiment_scores = []
        
        try:
            # Analyze price action for sentiment indicators
            for symbol in symbols[:2]:  # Limit to main symbols
                try:
                    if symbol.upper() in ['BTC', 'BITCOIN']:
                        ticker = yf.Ticker("BTC-USD")
                    elif symbol.upper() in ['ETH', 'ETHEREUM']:
                        ticker = yf.Ticker("ETH-USD")
                    else:
                        continue
                    
                    # Get recent price data
                    hist = ticker.history(period="7d", interval="1h")
                    
                    if len(hist) > 24:  # Ensure we have enough data
                        # Calculate sentiment based on price movements
                        recent_returns = hist['Close'].pct_change().dropna()
                        
                        # Trend sentiment
                        trend_sentiment = self._calculate_trend_sentiment(hist['Close'])
                        
                        # Volatility sentiment
                        volatility_sentiment = self._calculate_volatility_sentiment(recent_returns)
                        
                        # Volume sentiment
                        volume_sentiment = self._calculate_volume_sentiment(hist['Volume'])
                        
                        # Combine market data sentiments
                        combined_sentiment = (trend_sentiment + volatility_sentiment + volume_sentiment) / 3
                        
                        market_sentiment = SentimentScore(
                            compound=combined_sentiment,
                            positive=max(0, combined_sentiment),
                            negative=max(0, -combined_sentiment),
                            neutral=1 - abs(combined_sentiment),
                            confidence=0.7,  # Market data is generally reliable
                            source=f"market_data_{symbol}",
                            timestamp=datetime.utcnow(),
                            text_sample=f"Market data analysis for {symbol}"
                        )
                        
                        sentiment_scores.append(market_sentiment)
                
                except Exception as e:
                    logger.warning(f"Market data sentiment error for {symbol}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Market data sentiment analysis error: {e}")
        
        return sentiment_scores
    
    def _analyze_text_sentiment(self, text: str, source: str = "unknown") -> SentimentScore:
        """
        Analyze sentiment of a text using multiple methods and combine results
        """
        try:
            if not text or len(text.strip()) < 3:
                return SentimentScore(0.0, 0.0, 0.0, 1.0, 0.1, source, datetime.utcnow(), text[:100])
            
            # Clean text
            cleaned_text = self._clean_text(text)
            
            sentiments = []
            confidences = []
            
            # VADER sentiment
            try:
                vader_scores = self.vader_analyzer.polarity_scores(cleaned_text)
                sentiments.append(vader_scores['compound'])
                confidences.append(0.6)  # VADER is good for social media
            except:
                pass
            
            # TextBlob sentiment
            try:
                blob = TextBlob(cleaned_text)
                textblob_sentiment = blob.sentiment.polarity
                sentiments.append(textblob_sentiment)
                confidences.append(0.5)  # TextBlob is decent
            except:
                pass
            
            # FinBERT sentiment (financial text specialist)
            if self.finbert_available and len(cleaned_text) < 512:  # Token limit
                try:
                    finbert_result = self.finbert_pipeline(cleaned_text)[0]
                    
                    # Convert FinBERT labels to sentiment score
                    label = finbert_result['label'].lower()
                    confidence = finbert_result['score']
                    
                    if 'positive' in label:
                        finbert_sentiment = confidence
                    elif 'negative' in label:
                        finbert_sentiment = -confidence
                    else:  # neutral
                        finbert_sentiment = 0.0
                    
                    sentiments.append(finbert_sentiment)
                    confidences.append(0.8)  # FinBERT is specialized for finance
                    
                except Exception as e:
                    logger.debug(f"FinBERT analysis failed: {e}")
            
            # Flair sentiment
            if self.flair_available:
                try:
                    sentence = Sentence(cleaned_text)
                    self.flair_classifier.predict(sentence)
                    
                    flair_sentiment = 0.0
                    if sentence.labels:
                        label = sentence.labels[0]
                        confidence_score = label.score
                        
                        if label.value == 'POSITIVE':
                            flair_sentiment = confidence_score
                        elif label.value == 'NEGATIVE':
                            flair_sentiment = -confidence_score
                        
                        sentiments.append(flair_sentiment)
                        confidences.append(0.7)
                        
                except Exception as e:
                    logger.debug(f"Flair analysis failed: {e}")
            
            # Combine sentiments using weighted average
            if sentiments:
                total_weight = sum(confidences)
                weighted_sentiment = sum(s * c for s, c in zip(sentiments, confidences)) / total_weight
                combined_confidence = np.mean(confidences)
            else:
                # Fallback to simple keyword-based sentiment
                weighted_sentiment = self._simple_keyword_sentiment(cleaned_text)
                combined_confidence = 0.3
            
            # Calculate individual components
            positive = max(0, weighted_sentiment)
            negative = max(0, -weighted_sentiment)
            neutral = 1 - abs(weighted_sentiment)
            
            return SentimentScore(
                compound=weighted_sentiment,
                positive=positive,
                negative=negative,
                neutral=neutral,
                confidence=combined_confidence,
                source=source,
                timestamp=datetime.utcnow(),
                text_sample=text[:100] + "..." if len(text) > 100 else text
            )
            
        except Exception as e:
            logger.warning(f"Text sentiment analysis failed: {e}")
            return SentimentScore(0.0, 0.0, 0.0, 1.0, 0.1, source, datetime.utcnow(), text[:100])
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        try:
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove special characters but keep important punctuation
            text = re.sub(r'[^\w\s!?.,]', ' ', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Limit length for model constraints
            if len(text) > 500:
                text = text[:500] + "..."
            
            return text.strip()
            
        except:
            return text
    
    def _simple_keyword_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment as fallback"""
        positive_keywords = [
            'bullish', 'moon', 'pump', 'buy', 'long', 'bull', 'up', 'rise', 'gain',
            'profit', 'good', 'great', 'excellent', 'amazing', 'fantastic', 'positive',
            'growth', 'increase', 'surge', 'rally', 'breakthrough', 'success'
        ]
        
        negative_keywords = [
            'bearish', 'dump', 'sell', 'short', 'bear', 'down', 'fall', 'loss',
            'bad', 'terrible', 'awful', 'negative', 'decline', 'crash', 'drop',
            'fear', 'panic', 'disaster', 'failure', 'collapse', 'plummet'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        total_keywords = positive_count + negative_count
        
        if total_keywords == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total_keywords
        return max(-1, min(1, sentiment))
    
    def _calculate_trend_sentiment(self, prices: pd.Series) -> float:
        """Calculate sentiment based on price trend"""
        try:
            if len(prices) < 2:
                return 0.0
            
            # Calculate short-term and medium-term trends
            short_trend = (prices.iloc[-1] / prices.iloc[-6] - 1) if len(prices) >= 6 else 0
            medium_trend = (prices.iloc[-1] / prices.iloc[-24] - 1) if len(prices) >= 24 else 0
            
            # Combine trends with higher weight on recent
            trend_sentiment = (short_trend * 0.7 + medium_trend * 0.3) * 10  # Scale to sentiment range
            
            return max(-1, min(1, trend_sentiment))
            
        except:
            return 0.0
    
    def _calculate_volatility_sentiment(self, returns: pd.Series) -> float:
        """Calculate sentiment based on volatility (high volatility = fear = negative)"""
        try:
            if len(returns) < 5:
                return 0.0
            
            volatility = returns.std()
            
            # Normalize volatility to sentiment (high vol = negative sentiment)
            # Typical crypto volatility is around 0.03-0.05 (3-5%)
            normalized_vol = min(volatility / 0.05, 2.0)  # Cap at 2x normal vol
            
            # High volatility indicates fear/uncertainty
            vol_sentiment = -normalized_vol * 0.5  # Scale to -1 to 0
            
            return max(-1, min(0, vol_sentiment))
            
        except:
            return 0.0
    
    def _calculate_volume_sentiment(self, volumes: pd.Series) -> float:
        """Calculate sentiment based on volume patterns"""
        try:
            if len(volumes) < 10:
                return 0.0
            
            # Compare recent volume to historical average
            recent_volume = volumes.iloc[-3:].mean()  # Last 3 periods
            historical_volume = volumes.iloc[-20:-3].mean()  # Previous 17 periods
            
            if historical_volume > 0:
                volume_ratio = recent_volume / historical_volume
                
                # High volume with price increase is bullish, with price decrease is bearish
                # For now, just use volume increase as slight positive (confirmation)
                volume_sentiment = min((volume_ratio - 1) * 0.2, 0.3)
                
                return max(-0.3, min(0.3, volume_sentiment))
            
            return 0.0
            
        except:
            return 0.0
    
    def _aggregate_sentiment_scores(self, sentiment_scores: List[SentimentScore]) -> Dict:
        """Aggregate multiple sentiment scores into overall metrics"""
        try:
            if not sentiment_scores:
                return {
                    'overall': 0.0,
                    'confidence': 0.1,
                    'source_breakdown': {},
                    'social': 0.0,
                    'news': 0.0,
                    'volume_weighted': 0.0
                }
            
            # Group by source
            source_sentiments = defaultdict(list)
            source_confidences = defaultdict(list)
            
            for score in sentiment_scores:
                source_sentiments[score.source].append(score.compound)
                source_confidences[score.source].append(score.confidence)
            
            # Calculate source averages
            source_breakdown = {}
            for source, sentiments in source_sentiments.items():
                confidences = source_confidences[source]
                
                # Weighted average by confidence
                total_confidence = sum(confidences)
                if total_confidence > 0:
                    weighted_sentiment = sum(s * c for s, c in zip(sentiments, confidences)) / total_confidence
                    source_breakdown[source] = weighted_sentiment
                else:
                    source_breakdown[source] = np.mean(sentiments)
            
            # Calculate overall sentiment
            all_sentiments = [score.compound for score in sentiment_scores]
            all_confidences = [score.confidence for score in sentiment_scores]
            
            total_confidence = sum(all_confidences)
            if total_confidence > 0:
                overall_sentiment = sum(s * c for s, c in zip(all_sentiments, all_confidences)) / total_confidence
                average_confidence = np.mean(all_confidences)
            else:
                overall_sentiment = np.mean(all_sentiments)
                average_confidence = 0.5
            
            # Calculate category sentiments
            social_sources = [s for s in source_breakdown.keys() if 'reddit' in s.lower()]
            news_sources = [s for s in source_breakdown.keys() if any(x in s.lower() for x in ['news', 'yahoo', 'alpha'])]
            
            social_sentiment = np.mean([source_breakdown[s] for s in social_sources]) if social_sources else 0.0
            news_sentiment = np.mean([source_breakdown[s] for s in news_sources]) if news_sources else 0.0
            
            return {
                'overall': overall_sentiment,
                'confidence': average_confidence,
                'source_breakdown': source_breakdown,
                'social': social_sentiment,
                'news': news_sentiment,
                'volume_weighted': overall_sentiment  # Simplified for now
            }
            
        except Exception as e:
            logger.error(f"Error aggregating sentiment scores: {e}")
            return {
                'overall': 0.0,
                'confidence': 0.1,
                'source_breakdown': {},
                'social': 0.0,
                'news': 0.0,
                'volume_weighted': 0.0
            }
    
    def _determine_sentiment_trend(self, sentiment_score: float) -> str:
        """Determine sentiment trend from numerical score"""
        if sentiment_score > 0.1:
            return 'bullish'
        elif sentiment_score < -0.1:
            return 'bearish'
        else:
            return 'neutral'
    
    def get_sentiment_summary(self, hours: int = 24) -> Dict:
        """Get sentiment summary for the last N hours"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_sentiments = [
                s for s in self.sentiment_history 
                if s.timestamp >= cutoff_time
            ]
            
            if not recent_sentiments:
                return {'error': 'No recent sentiment data available'}
            
            # Calculate trends
            sentiment_values = [s.overall_sentiment for s in recent_sentiments]
            
            return {
                'current_sentiment': recent_sentiments[-1].overall_sentiment if recent_sentiments else 0.0,
                'average_sentiment': np.mean(sentiment_values),
                'sentiment_volatility': np.std(sentiment_values),
                'trend': 'improving' if len(sentiment_values) > 1 and sentiment_values[-1] > sentiment_values[0] else 'declining',
                'confidence': recent_sentiments[-1].confidence_level if recent_sentiments else 0.0,
                'data_points': len(recent_sentiments),
                'time_range_hours': hours
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {e}")
            return {'error': str(e)}
    
    def get_analyzer_status(self) -> Dict:
        """Get status of the sentiment analyzer"""
        return {
            'models_available': {
                'vader': True,
                'textblob': True,
                'finbert': self.finbert_available,
                'flair': self.flair_available
            },
            'apis_available': {
                'reddit': self.reddit_client is not None,
                'alpha_vantage': self.alpha_vantage_client is not None,
                'news_api': self.news_client is not None
            },
            'sentiment_history_size': len(self.sentiment_history),
            'cache_size': len(self.sentiment_cache)
        }


class NewsAnalyzer:
    """
    Specialized news analysis system
    """
    
    def __init__(self, news_api_key: Optional[str] = None):
        self.news_client = None
        
        if news_api_key:
            try:
                self.news_client = NewsApiClient(api_key=news_api_key)
            except:
                pass
        
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        
    async def get_market_news_sentiment(self, symbols: List[str], hours: int = 24) -> Dict:
        """Get news sentiment for specific market symbols"""
        try:
            news_articles = []
            
            # Get news from multiple sources
            if self.news_client:
                # Use News API
                for symbol in symbols:
                    query = f"{symbol} cryptocurrency bitcoin"
                    articles = self.news_client.get_everything(
                        q=query,
                        language='en',
                        sort_by='publishedAt',
                        from_param=(datetime.utcnow() - timedelta(hours=hours)).isoformat()
                    )
                    
                    news_articles.extend(articles.get('articles', []))
            
            # Analyze sentiment of news articles
            sentiment_scores = []
            
            for article in news_articles[:50]:  # Limit analysis
                title = article.get('title', '')
                description = article.get('description', '')
                
                if title or description:
                    text = f"{title} {description}"
                    sentiment = self.sentiment_analyzer._analyze_text_sentiment(text, "news")
                    sentiment_scores.append(sentiment)
            
            # Aggregate sentiment
            if sentiment_scores:
                overall_sentiment = np.mean([s.compound for s in sentiment_scores])
                confidence = np.mean([s.confidence for s in sentiment_scores])
            else:
                overall_sentiment = 0.0
                confidence = 0.0
            
            return {
                'overall_sentiment': overall_sentiment,
                'confidence': confidence,
                'articles_analyzed': len(sentiment_scores),
                'sentiment_distribution': {
                    'positive': len([s for s in sentiment_scores if s.compound > 0.1]),
                    'neutral': len([s for s in sentiment_scores if abs(s.compound) <= 0.1]),
                    'negative': len([s for s in sentiment_scores if s.compound < -0.1])
                }
            }
            
        except Exception as e:
            logger.error(f"News analysis error: {e}")
            return {'error': str(e)}


class SocialMediaAnalyzer:
    """
    Specialized social media sentiment analysis
    """
    
    def __init__(self, reddit_credentials: Optional[Dict] = None):
        self.reddit_client = None
        
        if reddit_credentials:
            try:
                self.reddit_client = praw.Reddit(**reddit_credentials)
            except:
                pass
        
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        
    async def analyze_social_sentiment(self, symbols: List[str]) -> Dict:
        """Analyze social media sentiment for symbols"""
        try:
            if not self.reddit_client:
                return {'error': 'Reddit client not available'}
            
            sentiment_data = await self.sentiment_analyzer._analyze_reddit_sentiment(symbols)
            
            # Process and return summary
            if sentiment_data:
                overall_sentiment = np.mean([s.compound for s in sentiment_data])
                confidence = np.mean([s.confidence for s in sentiment_data])
                
                return {
                    'overall_sentiment': overall_sentiment,
                    'confidence': confidence,
                    'posts_analyzed': len(sentiment_data),
                    'sentiment_trend': self.sentiment_analyzer._determine_sentiment_trend(overall_sentiment)
                }
            else:
                return {
                    'overall_sentiment': 0.0,
                    'confidence': 0.0,
                    'posts_analyzed': 0,
                    'sentiment_trend': 'neutral'
                }
                
        except Exception as e:
            logger.error(f"Social media analysis error: {e}")
            return {'error': str(e)}