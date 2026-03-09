"""
News2Trade AI - Pipeline Orchestrator
Ties together all components into a single analysis pipeline.
"""
import pandas as pd
from news_fetcher import fetch_news, get_sample_news, fetch_scraped_news, fetch_news_from_api
from sentiment_analyzer import SentimentAnalyzer
from hype_detector import HypeDetector
from trading_model import TradingSignalModel


class News2TradePipeline:
    """
    End-to-end pipeline:
        News API → Sentiment Analysis → Hype Detection → ML Signal → Explanation
    """

    def __init__(self, use_finbert: bool = True):
        print("=" * 60)
        print("  News2Trade AI — Initializing Pipeline")
        print("=" * 60)

        self.sentiment_analyzer = SentimentAnalyzer(use_finbert=use_finbert)
        self.hype_detector = HypeDetector()
        self.trading_model = TradingSignalModel()

        # Try loading saved model
        self.trading_model.load_model()

    # ─── Full Pipeline ────────────────────────────────────────────

    def analyze_news_feed(
        self,
        query: str = "stock market finance",
        page_size: int = 30,
        beginner_mode: bool = False,
        use_api: bool = True,
        use_scraping: bool = True,
        enrich_bodies: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch news & run the full pipeline on all articles.
        Returns enriched DataFrame with sentiment, hype, and signal columns.
        """
        # 1. Fetch news from selected sources
        df = fetch_news(
            query=query,
            page_size=page_size,
            use_api=use_api,
            use_scraping=use_scraping,
            enrich_bodies=enrich_bodies,
        )
        return self._process_dataframe(df, beginner_mode)

    def analyze_single_article(
        self,
        title: str,
        text: str,
        source: str = "",
        beginner_mode: bool = False,
    ) -> dict:
        """
        Analyze a single user-provided article.
        """
        full_text = f"{title}. {text}" if title else text

        # Sentiment
        sentiment = self.sentiment_analyzer.analyze(full_text)

        # Hype detection
        hype = self.hype_detector.detect(full_text, source)

        # Trading signal
        signal = self.trading_model.predict(
            text=full_text,
            title=title,
            sentiment_result=sentiment,
            hype_result=hype,
            beginner_mode=beginner_mode,
        )

        return {
            "title": title,
            "text": text,
            "source": source,
            "sentiment": sentiment,
            "hype": hype,
            "signal": signal,
        }

    def train_model(self, query: str = "stock market finance", page_size: int = 30):
        """
        Fetch news, analyze, and train the ML model.
        """
        print("\n[PIPELINE] Training model on news dataset...")
        df = fetch_news(query, page_size)

        if df.empty:
            print("[WARN] No data to train on.")
            return

        # Analyze all articles
        sentiment_results = []
        hype_results = []

        for _, row in df.iterrows():
            text = str(row.get("text", ""))
            source = str(row.get("source", ""))

            sent = self.sentiment_analyzer.analyze(text)
            hype = self.hype_detector.detect(text, source)

            sentiment_results.append(sent)
            hype_results.append(hype)

        # Train
        self.trading_model.train_on_dataset(df, sentiment_results, hype_results)
        print("[PIPELINE] Training complete.\n")

    # ─── Internal ─────────────────────────────────────────────────

    def _process_dataframe(self, df: pd.DataFrame, beginner_mode: bool) -> pd.DataFrame:
        """Run sentiment + hype + signal on each row."""
        if df.empty:
            return df

        # If model isn't trained, train it first on this data
        if not self.trading_model.is_trained:
            print("[PIPELINE] Model not trained. Training on current dataset...")
            sentiment_results = []
            hype_results = []
            for _, row in df.iterrows():
                text = str(row.get("text", ""))
                source = str(row.get("source", ""))
                sent = self.sentiment_analyzer.analyze(text)
                hype = self.hype_detector.detect(text, source)
                sentiment_results.append(sent)
                hype_results.append(hype)
            self.trading_model.train_on_dataset(df, sentiment_results, hype_results)

        # Now predict for each article
        results = []
        for _, row in df.iterrows():
            text = str(row.get("text", ""))
            title = str(row.get("title", ""))
            source = str(row.get("source", ""))

            sent = self.sentiment_analyzer.analyze(text)
            hype = self.hype_detector.detect(text, source)
            signal = self.trading_model.predict(
                text=text,
                title=title,
                sentiment_result=sent,
                hype_result=hype,
                beginner_mode=beginner_mode,
            )

            results.append({
                "sentiment_score": sent["sentiment_score"],
                "sentiment_label": sent["sentiment_label"],
                "sentiment_confidence": sent["confidence"],
                "sentiment_method": sent["method"],
                "hype_score": hype["hype_score"],
                "is_hype": hype["is_hype"],
                "hype_flags": "; ".join(hype["flags"]) if hype["flags"] else "",
                "signal": signal["signal"],
                "signal_confidence": signal["confidence"],
                "risk_level": signal["risk_level"],
                "beginner_override": signal["beginner_override"],
                "explanation": signal["explanation"],
            })

        result_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), result_df], axis=1)
