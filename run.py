"""
News2Trade AI — Quick CLI Runner
Run this file directly to analyze news from the terminal without the browser UI.

Usage:
    python run.py                        # defaults: "stock market", 10 articles
    python run.py "Tesla" 15             # custom query + article count
    python run.py "Bitcoin crash" 5      # any search term
"""
import sys
from pipeline import News2TradePipeline


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "stock market"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print()
    print("=" * 60)
    print("  📈  News2Trade AI — CLI Mode")
    print("=" * 60)
    print(f"  Query   : {query}")
    print(f"  Articles: {count}")
    print(f"  Mode    : Beginner Safety ON")
    print("=" * 60)
    print()

    # Initialize
    pipeline = News2TradePipeline(use_finbert=False)

    # Train if needed (fast, uses VADER)
    if not pipeline.trading_model.is_trained:
        print("[*] Training model on live data...\n")
        pipeline.train_model(query=query, page_size=count)

    # Analyze
    print("[*] Analyzing news feed...\n")
    df = pipeline.analyze_news_feed(
        query=query,
        page_size=count,
        beginner_mode=True,
        use_api=True,
        use_scraping=True,
    )

    if df.empty:
        print("No articles found. Try a different query.")
        return

    # ── Results ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  RESULTS — {len(df)} articles analyzed")
    print("=" * 60)
    print()

    # Signal summary
    buy  = (df["signal"] == "Buy").sum()
    sell = (df["signal"] == "Sell").sum()
    hold = (df["signal"] == "Hold").sum()
    hype = int(df["is_hype"].sum()) if "is_hype" in df.columns else 0

    print(f"  🟢 Buy : {buy:3d}    🔴 Sell: {sell:3d}    🟡 Hold: {hold:3d}    ⚠️  Hype: {hype}")
    print()

    # Per-article table
    print("-" * 100)
    print(f"  {'Signal':<8} {'Conf':>5}  {'Sent':>6}  {'Hype':>5}  {'Risk':<7}  Title")
    print("-" * 100)

    for _, row in df.iterrows():
        signal = row.get("signal", "?")
        conf   = row.get("signal_confidence", 0)
        sent   = row.get("sentiment_score", 0)
        hype_s = row.get("hype_score", 0)
        risk   = row.get("risk_level", "?")
        title  = str(row.get("title", ""))[:65]

        icon = {"Buy": "🟢", "Sell": "🔴", "Hold": "🟡"}.get(signal, "  ")
        print(f"  {icon} {signal:<5} {conf:5.0%}  {sent:+.3f}  {hype_s:5.0%}  {risk:<7}  {title}")

    print("-" * 100)

    # Save to CSV
    out_file = "results.csv"
    cols = [c for c in ["title", "source", "data_source", "signal", "signal_confidence",
                         "sentiment_score", "hype_score", "is_hype", "risk_level",
                         "explanation", "url"] if c in df.columns]
    df[cols].to_csv(out_file, index=False)
    print(f"\n  📄 Full results saved to {out_file}")

    # Overall recommendation
    print()
    if buy > sell and buy > hold:
        print("  📊 Overall Sentiment: BULLISH — More buy signals detected.")
    elif sell > buy and sell > hold:
        print("  📊 Overall Sentiment: BEARISH — More sell signals detected.")
    else:
        print("  📊 Overall Sentiment: NEUTRAL — Market signals are mixed.")
    print()


if __name__ == "__main__":
    main()
