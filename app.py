#!/usr/bin/env python3
"""
News2TradeAI — Main Entry Point
================================
Usage
-----
    python app.py train          # Build dataset, train & evaluate all models
    python app.py predict "..."  # Predict impact of a headline
    python app.py api            # Start the FastAPI prediction server
    python app.py dashboard      # Launch the Streamlit dashboard
    python app.py demo           # Generate demo dataset + train + launch dashboard
"""

import sys
import argparse
import logging

# ── Setup ──────────────────────────────────────────────────────────
sys.path.insert(0, ".")

from utils.logger import setup_logging

logger = setup_logging()


def cmd_train(args):
    """Build dataset, preprocess, engineer features, train & benchmark."""
    from data.dataset_builder import load_dataset, create_demo_dataset
    from preprocessing.text_pipeline import preprocess_dataframe
    from preprocessing.feature_engineering import build_feature_matrix
    from training.train_pipeline import run_benchmark
    import joblib
    from config.settings import MODEL_DIR

    logger.info("═" * 60)
    logger.info("  News2TradeAI — Training Pipeline")
    logger.info("═" * 60)

    # 1. Dataset - Generate 5000 samples (3000 train + 2000 test)
    logger.info("Step 1/4 — Loading dataset…")
    try:
        df = load_dataset()
        if df.empty:
            raise FileNotFoundError
    except Exception:
        logger.info("No real dataset found — generating demo dataset (5000 samples).")
        df = create_demo_dataset(n_samples=5000)

    logger.info(f"Dataset size: {len(df)} samples (2000 reserved for testing)")

    # 2. Preprocessing
    logger.info("Step 2/4 — Preprocessing text…")
    df = preprocess_dataframe(df, text_col="title")

    # 3. Feature engineering
    logger.info("Step 3/4 — Building feature matrix…")
    X, y = build_feature_matrix(df, text_col="clean_text", use_finbert=False)

    # Save TF-IDF vectorizer for inference
    from preprocessing.feature_engineering import TextFeatureExtractor

    vec = TextFeatureExtractor()
    vec.fit_transform(df["clean_text"])
    joblib.dump(vec.vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")

    # 4. Training & benchmarking (uses 2000 sample test set internally)
    logger.info("Step 4/4 — Training models…")
    summary = run_benchmark(X, y, tune=not args.no_tune, test_size=2000)
    print("\n" + summary.to_string(index=False))
    logger.info("Training complete ✓")


def cmd_predict(args):
    """Single headline prediction via CLI."""
    from api.prediction_pipeline import PredictionPipeline, EnsemblePipeline

    if args.ensemble:
        # Use all models ensemble
        pipeline = EnsemblePipeline(beginner_mode=args.beginner)
        result = pipeline.predict(args.headline)
        
        print(f"\n{'═' * 60}")
        print(f"  📰 Headline : {result['headline']}")
        print(f"{'─' * 60}")
        print(f"  🎯 Prediction : {result['prediction']}")
        print(f"  📊 Confidence : {result['confidence']:.1%}")
        print(f"  🤝 Agreement  : {result['agreement']:.0%} ({len(result['models_used'])} models)")
        print(f"{'─' * 60}")
        print(f"  💡 Action: {result['action']}")
        if result['safety_applied']:
            print(f"  ⚠️  Safety: {result['safety_reason']}")
        print(f"{'─' * 60}")
        print(f"  Probabilities : {result['probabilities']}")
        print(f"  Model Votes   : {result['model_votes']}")
        print(f"  Individual    : {result['individual_predictions']}")
        print(f"  Sentiment     : {result['sentiment']}")
        if args.beginner:
            print(f"{'─' * 60}")
            print(f"  🔰 BEGINNER MODE: Requires ≥70% confidence & ≥60% model agreement")
        print(f"{'═' * 60}\n")
    else:
        # Single model
        pipeline = PredictionPipeline(args.model)
        result = pipeline.predict(args.headline)
        print(f"\n{'═' * 50}")
        print(f"  Headline : {result['headline']}")
        print(f"  Prediction : {result['prediction']}")
        print(f"  Confidence : {result['confidence']:.1%}")
        print(f"  Probabilities : {result['probabilities']}")
        print(f"  Sentiment : {result['sentiment']}")
        print(f"  Keywords : {result['financial_keywords']}")
        print(f"{'═' * 50}\n")


def cmd_api(_args):
    """Launch FastAPI server."""
    from api.server import start_api

    logger.info("Starting API server…")
    start_api()


def cmd_dashboard(_args):
    """Launch Streamlit dashboard."""
    import subprocess

    logger.info("Launching Streamlit dashboard…")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.headless", "true"],
        check=True,
    )


def cmd_demo(args):
    """Full demo: generate data → train → launch dashboard."""
    logger.info("Running full demo pipeline…")

    # Train first
    args.no_tune = True
    cmd_train(args)

    # Then launch dashboard
    cmd_dashboard(args)


# ── CLI ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        prog="News2TradeAI",
        description="Financial News Impact Predictor — ML Research System",
    )
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train all models")
    p_train.add_argument("--no-tune", action="store_true",
                         help="Skip hyper-parameter tuning for faster runs")

    # predict
    p_pred = sub.add_parser("predict", help="Predict from headline")
    p_pred.add_argument("headline", type=str, help="News headline text")
    p_pred.add_argument("--model", default="xgboost", help="Model name (ignored if --ensemble)")
    p_pred.add_argument("--ensemble", "-e", action="store_true",
                        help="Use all models and aggregate predictions")
    p_pred.add_argument("--beginner", "-b", action="store_true",
                        help="Enable beginner/safe mode (requires higher confidence)")

    # api
    sub.add_parser("api", help="Start REST API server")

    # dashboard
    sub.add_parser("dashboard", help="Launch Streamlit dashboard")

    # demo
    sub.add_parser("demo", help="Full demo: dataset → train → dashboard")

    args = parser.parse_args()

    commands = {
        "train": cmd_train,
        "predict": cmd_predict,
        "api": cmd_api,
        "dashboard": cmd_dashboard,
        "demo": cmd_demo,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
