"""Quick diagnostic: test predictions across diverse headlines."""
from api.prediction_pipeline import PredictionPipeline

pipe = PredictionPipeline("random_forest")

headlines = [
    # ── BULLISH (should predict UP) ──
    "Apple reports record quarterly earnings beating Wall Street expectations",
    "NVIDIA soars on stellar AI chip revenue topping all estimates",
    "Microsoft profit margins expand as cloud growth accelerates",
    "Tesla rebounds strongly gaining momentum with record deliveries",
    "Amazon tops revenue estimates as AWS business booms",
    "JPMorgan upgrades tech sector to strong buy outperform",
    "Netflix subscriber gains surge past analyst forecasts",
    "Google beats estimates with impressive quarterly results",
    # ── BEARISH (should predict DOWN) ──
    "Tesla stock plunges 15% after disappointing revenue guidance",
    "Goldman Sachs announces massive layoffs amid recession fears",
    "Meta faces class action lawsuit and regulatory investigation",
    "Amazon warns of significant revenue shortfall next quarter",
    "NVIDIA credit rating downgraded amid declining chip demand",
    "Apple misses earnings estimates as iPhone sales collapse",
    "Market crash feared as recession indicators flash red",
    "Microsoft slashes workforce cutting 10000 jobs amid weak outlook",
    # ── NEUTRAL (should predict NEUTRAL) ──
    "Federal Reserve maintains current policy amid mixed economic signals",
    "Apple holds annual shareholder meeting with no major announcements",
    "Google reports steady growth in line with market forecasts",
    "Tesla trading volume remains average ahead of earnings report",
]

print(f"{'PRED':8s} {'CONF':>5s}  {'DOWN':>5s} {'NEUT':>5s} {'UP':>5s}  HEADLINE")
print("-" * 110)
for h in headlines:
    r = pipe.predict(h)
    p = r["probabilities"]
    print(
        f"{r['prediction']:8s} {r['confidence']:5.1%}  "
        f"{p['DOWN']:5.2f} {p['NEUTRAL']:5.2f} {p['UP']:5.2f}  "
        f"{h[:70]}"
    )
