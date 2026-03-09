"""
FastAPI Prediction Server
==========================
RESTful API that exposes the prediction pipeline, stock data, and
news sentiment endpoints.
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import API_HOST, API_PORT

logger = logging.getLogger(__name__)

app = FastAPI(
    title="News2TradeAI — Financial News Impact Predictor",
    description="Predict stock price movement from news headlines using ML.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response schemas ──────────────────────────────────
class PredictionRequest(BaseModel):
    headline: str = Field(..., min_length=5, max_length=1000,
                          description="Financial news headline or article text.")
    model_name: Optional[str] = Field("xgboost", description="Model to use.")


class PredictionResponse(BaseModel):
    headline: str
    prediction: str
    confidence: float
    probabilities: dict
    sentiment: dict
    financial_keywords: list
    entities: list
    model_used: str


class BatchRequest(BaseModel):
    headlines: list[str]
    model_name: Optional[str] = "xgboost"


class HealthResponse(BaseModel):
    status: str
    version: str


# ─── Endpoints ────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    """Predict stock movement from a single headline."""
    try:
        from api.prediction_pipeline import get_pipeline

        pipeline = get_pipeline(req.model_name)
        result = pipeline.predict(req.headline)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch")
async def predict_batch(req: BatchRequest):
    """Batch prediction for multiple headlines."""
    try:
        from api.prediction_pipeline import get_pipeline

        pipeline = get_pipeline(req.model_name)
        results = pipeline.predict_batch(req.headlines)
        return {"predictions": results}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stock/{ticker}")
async def get_stock_data(ticker: str, period: str = "1mo"):
    """Fetch recent stock data for a ticker."""
    try:
        from data.stock_data import fetch_yahoo_history
        from datetime import datetime, timedelta

        period_map = {"1w": 7, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}
        days = period_map.get(period, 30)
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = fetch_yahoo_history(ticker, start=start)
        return df.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stock/search/{query}")
async def search_ticker(query: str, limit: int = 10):
    """Search for stock tickers by company name or symbol via yfinance."""
    try:
        import yfinance as yf

        results = []
        # yfinance search (works for well-known tickers)
        tck = yf.Ticker(query.upper())
        info = tck.info or {}
        if info.get("symbol"):
            results.append({
                "symbol": info["symbol"],
                "name": info.get("shortName", info.get("longName", query.upper())),
                "exchange": info.get("exchange", ""),
                "type": info.get("quoteType", ""),
            })
        return {"query": query, "results": results[:limit]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Run ──────────────────────────────────────────────────────────
def start_api():
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    start_api()
