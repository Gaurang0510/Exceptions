"""
News2Trade AI - Quick validation test
"""
from news_fetcher import get_sample_news
from sentiment_analyzer import SentimentAnalyzer
from hype_detector import HypeDetector
from trading_model import TradingSignalModel
from pipeline import News2TradePipeline

# Quick test
print("=== Testing News Fetcher ===")
df = get_sample_news()
print(f"Sample articles: {len(df)}")

print("\n=== Testing Sentiment Analyzer (VADER) ===")
sa = SentimentAnalyzer(use_finbert=False)
result = sa.analyze("Apple Reports Record Q4 Earnings, Beating Analyst Expectations")
print(f"Sentiment: {result['sentiment_label']} ({result['sentiment_score']:+.2f}), Confidence: {result['confidence']:.2f}")

result2 = sa.analyze("Global Markets Plunge as Economy Shows Signs of Deflation")
print(f"Sentiment: {result2['sentiment_label']} ({result2['sentiment_score']:+.2f}), Confidence: {result2['confidence']:.2f}")

print("\n=== Testing Hype Detector ===")
hd = HypeDetector()
hype1 = hd.detect("GUARANTEED 100x Returns! This Secret Stock Will Make You a Millionaire!", "Unknown Blog")
print(f"Hype: {hype1['is_hype']} (score: {hype1['hype_score']:.2f})")

hype2 = hd.detect("Apple Reports Record Q4 Earnings", "Reuters")
print(f"Hype: {hype2['is_hype']} (score: {hype2['hype_score']:.2f})")

print("\n=== Testing Full Pipeline ===")
pipeline = News2TradePipeline(use_finbert=False)
result = pipeline.analyze_single_article(
    title="Tesla Stock Surges 12% After Announcing New Gigafactory",
    text="Tesla shares rallied after CEO confirmed plans for a 5B manufacturing plant.",
    source="Bloomberg",
    beginner_mode=True,
)
sig = result["signal"]
print(f"Signal: {sig['signal']} (Confidence: {sig['confidence']:.0%}, Risk: {sig['risk_level']})")

# Test batch analysis
print("\n=== Testing Batch Analysis ===")
batch_df = pipeline.analyze_news_feed(beginner_mode=True)
print(f"Batch analyzed: {len(batch_df)} articles")
if "signal" in batch_df.columns:
    print(f"Signal distribution:\n{batch_df['signal'].value_counts().to_string()}")

print("\n=== ALL TESTS PASSED ===")
