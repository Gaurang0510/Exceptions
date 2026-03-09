"""
News2Trade AI - Streamlit Web Application
Interactive UI for sentiment-driven trading signals.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pipeline import News2TradePipeline

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="News2Trade AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .signal-buy {
        background-color: #0d6e3b;
        color: white;
        padding: 12px 24px;
        border-radius: 10px;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
    }
    .signal-sell {
        background-color: #c0392b;
        color: white;
        padding: 12px 24px;
        border-radius: 10px;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
    }
    .signal-hold {
        background-color: #f39c12;
        color: white;
        padding: 12px 24px;
        border-radius: 10px;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
    }
    .metric-card {
        background-color: #1e1e2e;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
    }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low { color: #2ecc71; font-weight: bold; }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Pipeline (cached) ────────────────────────────────
@st.cache_resource(show_spinner="Loading News2Trade AI models...")
def load_pipeline():
    return News2TradePipeline(use_finbert=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">📈 News2Trade AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Explainable, Sentiment-Driven Trading Signals</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/stock-market.png", width=80)
        st.title("⚙️ Settings")
        st.divider()

        beginner_mode = st.toggle(
            "🛡️ Beginner Safety Mode",
            value=True,
            help="Filters risky low-confidence signals and forces them to HOLD",
        )

        st.divider()
        st.markdown("### 📡 Data Sources")
        use_api = st.toggle("NewsAPI (Live)", value=True, help="Fetch news from NewsAPI using your API key")
        use_scraping = st.toggle("Web Scraping", value=True, help="Scrape Reuters, CNBC, MarketWatch, Yahoo Finance, Google News")
        enrich_bodies = st.toggle("Enrich Headlines", value=False, help="Scrape full article body text (slower but more accurate)")

        st.divider()
        mode = st.radio(
            "Analysis Mode",
            ["📰 Analyze News Feed", "✍️ Custom Article Input"],
            index=0,
        )

        st.divider()
        st.markdown("### ℹ️ About")
        st.markdown(
            "**News2Trade AI** converts financial news into "
            "actionable trading signals using NLP and ML."
        )
        st.markdown("---")
        st.caption("⚠️ Not financial advice. For educational use only.")

    # Load pipeline
    pipeline = load_pipeline()

    if mode == "📰 Analyze News Feed":
        render_feed_mode(pipeline, beginner_mode, use_api, use_scraping, enrich_bodies)
    else:
        render_custom_mode(pipeline, beginner_mode)


# ─── Mode 1: News Feed Analysis ──────────────────────────────────

def render_feed_mode(pipeline, beginner_mode, use_api=True, use_scraping=True, enrich_bodies=False):
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "🔍 Search financial news",
            value="stock market",
            placeholder="e.g., Tesla earnings, Bitcoin, Fed interest rate",
        )
    with col2:
        page_size = st.slider("Articles", 10, 50, 25)

    # Show active data sources
    sources = []
    if use_api:
        sources.append("NewsAPI")
    if use_scraping:
        sources.append("Web Scraping (Reuters, CNBC, MarketWatch, Yahoo Finance, Google News)")
    if not sources:
        sources.append("Sample Dataset (fallback)")
    st.caption(f"📡 Active sources: {', '.join(sources)}")

    if st.button("🚀 Analyze News", type="primary", width='stretch'):
        with st.spinner("Fetching and analyzing news..."):
            df = pipeline.analyze_news_feed(
                query=query,
                page_size=page_size,
                beginner_mode=beginner_mode,
                use_api=use_api,
                use_scraping=use_scraping,
                enrich_bodies=enrich_bodies,
            )

        if df.empty:
            st.error("No articles found. Try a different search query.")
            return

        st.session_state["feed_df"] = df
        st.success(f"✅ Analyzed {len(df)} articles!")

    if "feed_df" in st.session_state:
        df = st.session_state["feed_df"]
        render_dashboard(df, beginner_mode)


def render_dashboard(df: pd.DataFrame, beginner_mode: bool):
    """Render the full analysis dashboard."""

    # ── Summary Metrics ────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Dashboard Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    signal_counts = df["signal"].value_counts()
    with col1:
        st.metric("Total Articles", len(df))
    with col2:
        st.metric("🟢 Buy Signals", signal_counts.get("Buy", 0))
    with col3:
        st.metric("🔴 Sell Signals", signal_counts.get("Sell", 0))
    with col4:
        st.metric("🟡 Hold Signals", signal_counts.get("Hold", 0))
    with col5:
        hype_count = df["is_hype"].sum() if "is_hype" in df.columns else 0
        st.metric("⚠️ Hype Flagged", int(hype_count))

    # ── Tabs ───────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Signal Table", "📈 Visualizations", "🔍 Article Deep Dive", "🧠 Model Insights"
    ])

    # TAB 1: Signal Table
    with tab1:
        render_signal_table(df)

    # TAB 2: Visualizations
    with tab2:
        render_visualizations(df)

    # TAB 3: Article Deep Dive
    with tab3:
        render_article_deep_dive(df)

    # TAB 4: Model Insights
    with tab4:
        render_model_insights(df)


def render_signal_table(df: pd.DataFrame):
    """Display the article table with signals."""
    st.subheader("📋 Trading Signals for All Articles")

    display_cols = ["title", "source", "data_source", "signal", "signal_confidence",
                    "sentiment_label", "sentiment_score", "hype_score",
                    "is_hype", "risk_level"]
    available_cols = [c for c in display_cols if c in df.columns]

    # Color-code signals
    def style_signal(val):
        colors = {"Buy": "#0d6e3b", "Sell": "#c0392b", "Hold": "#f39c12"}
        return f"background-color: {colors.get(val, '#333')}; color: white; font-weight: bold"

    def style_risk(val):
        colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
        return f"color: {colors.get(val, '#fff')}; font-weight: bold"

    styled_df = df[available_cols].style.map(
        style_signal, subset=["signal"] if "signal" in available_cols else []
    ).map(
        style_risk, subset=["risk_level"] if "risk_level" in available_cols else []
    )

    st.dataframe(styled_df, width='stretch', height=500)

    # Download
    csv = df[available_cols].to_csv(index=False)
    st.download_button("📥 Download Results (CSV)", csv, "news2trade_results.csv", "text/csv")


def render_visualizations(df: pd.DataFrame):
    """Create charts and visualizations."""
    col1, col2 = st.columns(2)

    with col1:
        # Signal distribution pie chart
        if "signal" in df.columns:
            signal_counts = df["signal"].value_counts().reset_index()
            signal_counts.columns = ["Signal", "Count"]
            fig = px.pie(
                signal_counts, names="Signal", values="Count",
                title="Signal Distribution",
                color="Signal",
                color_discrete_map={"Buy": "#2ecc71", "Sell": "#e74c3c", "Hold": "#f39c12"},
                hole=0.4,
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, width='stretch')

    with col2:
        # Sentiment score distribution
        if "sentiment_score" in df.columns:
            fig = px.histogram(
                df, x="sentiment_score", nbins=20,
                title="Sentiment Score Distribution",
                color_discrete_sequence=["#00C9FF"],
                labels={"sentiment_score": "Sentiment Score"},
            )
            fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, width='stretch')

    col3, col4 = st.columns(2)

    with col3:
        # Hype score vs Sentiment scatter
        if "hype_score" in df.columns and "sentiment_score" in df.columns:
            fig = px.scatter(
                df, x="sentiment_score", y="hype_score",
                color="signal" if "signal" in df.columns else None,
                size="signal_confidence" if "signal_confidence" in df.columns else None,
                hover_data=["title"] if "title" in df.columns else None,
                title="Sentiment vs Hype Score",
                color_discrete_map={"Buy": "#2ecc71", "Sell": "#e74c3c", "Hold": "#f39c12"},
                labels={"sentiment_score": "Sentiment", "hype_score": "Hype Score"},
            )
            fig.add_hline(y=0.65, line_dash="dash", line_color="red", opacity=0.5,
                          annotation_text="Hype Threshold")
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, width='stretch')

    with col4:
        # Confidence distribution by signal
        if "signal_confidence" in df.columns and "signal" in df.columns:
            fig = px.box(
                df, x="signal", y="signal_confidence",
                color="signal",
                title="Confidence by Signal Type",
                color_discrete_map={"Buy": "#2ecc71", "Sell": "#e74c3c", "Hold": "#f39c12"},
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, width='stretch')

    # Sentiment trend bar chart (by source)
    if "source" in df.columns and "sentiment_score" in df.columns:
        source_sent = df.groupby("source")["sentiment_score"].mean().sort_values()
        fig = px.bar(
            x=source_sent.values, y=source_sent.index,
            orientation="h",
            title="Average Sentiment by Source",
            labels={"x": "Avg Sentiment Score", "y": "Source"},
            color=source_sent.values,
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(template="plotly_dark", height=max(300, len(source_sent) * 30))
        st.plotly_chart(fig, width='stretch')


def render_article_deep_dive(df: pd.DataFrame):
    """Let users pick an article and see full analysis."""
    st.subheader("🔍 Deep Dive into Individual Article")

    if "title" not in df.columns:
        st.info("No articles to inspect.")
        return

    # Article selector
    titles = df["title"].tolist()
    selected_title = st.selectbox("Select an article:", titles)
    row = df[df["title"] == selected_title].iloc[0]

    # Signal badge
    signal = row.get("signal", "Hold")
    badge_class = f"signal-{signal.lower()}"
    st.markdown(
        f'<div class="{badge_class}">🎯 Signal: {signal}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence", f"{row.get('signal_confidence', 0):.0%}")
    with col2:
        st.metric("Sentiment", f"{row.get('sentiment_score', 0):+.2f}")
    with col3:
        risk = row.get("risk_level", "Unknown")
        st.metric("Risk Level", risk)

    # Full explanation
    st.markdown("---")
    st.markdown("### 📝 Full Explanation")
    explanation = row.get("explanation", "No explanation available.")
    st.text(explanation)

    # Article details
    with st.expander("📰 Article Details"):
        st.markdown(f"**Source:** {row.get('source', 'Unknown')}")
        st.markdown(f"**Published:** {row.get('published_at', 'N/A')}")
        if row.get("url"):
            st.markdown(f"**URL:** [{row['url']}]({row['url']})")
        st.markdown(f"**Full Text:** {row.get('text', 'N/A')}")

    # Hype analysis
    with st.expander("⚠️ Hype Analysis"):
        hype_score = row.get("hype_score", 0)
        is_hype = row.get("is_hype", False)

        if is_hype:
            st.error(f"🚨 HYPE DETECTED — Score: {hype_score:.0%}")
        else:
            st.success(f"✅ Not flagged as hype — Score: {hype_score:.0%}")

        # Hype gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=hype_score * 100,
            title={"text": "Hype Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e74c3c" if is_hype else "#2ecc71"},
                "steps": [
                    {"range": [0, 40], "color": "#1a472a"},
                    {"range": [40, 65], "color": "#7d6608"},
                    {"range": [65, 100], "color": "#78281f"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 65,
                },
            },
        ))
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, width='stretch')

        if row.get("hype_flags"):
            st.markdown("**Flags:**")
            for flag in str(row["hype_flags"]).split(";"):
                if flag.strip():
                    st.markdown(f"  • {flag.strip()}")


def render_model_insights(df: pd.DataFrame):
    """Show model performance and feature importances."""
    st.subheader("🧠 Model Insights & Explainability")

    col1, col2 = st.columns(2)

    with col1:
        # Feature importance (if available from pipeline)
        st.markdown("### Feature Importances")
        st.caption("What the ML model considers most important")

        # We'll extract from the pipeline's model
        try:
            pipeline = load_pipeline()
            importances = pipeline.trading_model.get_feature_importances()
            if importances:
                imp_df = pd.DataFrame([
                    {"Feature": k, "Importance": v}
                    for k, v in sorted(importances.items(), key=lambda x: -x[1])
                ])
                fig = px.bar(
                    imp_df, x="Importance", y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Viridis",
                    title="Feature Importance (Random Forest)",
                )
                fig.update_layout(template="plotly_dark", height=450)
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("Feature importances will appear after the model is trained.")
        except Exception as e:
            st.info(f"Train the model first to see feature importances.")

    with col2:
        st.markdown("### Signal Confidence Distribution")
        if "signal_confidence" in df.columns:
            fig = px.histogram(
                df, x="signal_confidence", nbins=15,
                title="How confident is the model?",
                color_discrete_sequence=["#92FE9D"],
            )
            fig.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig, width='stretch')

    # Model summary stats
    st.markdown("### 📊 Summary Statistics")
    summary = {
        "Total Articles Analyzed": len(df),
        "Average Sentiment Score": f"{df['sentiment_score'].mean():+.3f}" if "sentiment_score" in df.columns else "N/A",
        "Average Confidence": f"{df['signal_confidence'].mean():.1%}" if "signal_confidence" in df.columns else "N/A",
        "Hype Articles Detected": int(df["is_hype"].sum()) if "is_hype" in df.columns else 0,
        "Beginner Overrides": int(df["beginner_override"].sum()) if "beginner_override" in df.columns else 0,
        "Sentiment Method": df["sentiment_method"].iloc[0] if "sentiment_method" in df.columns else "N/A",
        "Data Sources Used": ", ".join(df["data_source"].unique()) if "data_source" in df.columns else "N/A",
    }
    summary_df = pd.DataFrame([{"Metric": k, "Value": str(v)} for k, v in summary.items()])
    st.table(summary_df)


# ─── Mode 2: Custom Article Input ────────────────────────────────

def render_custom_mode(pipeline, beginner_mode):
    st.subheader("✍️ Analyze Your Own News Article")
    st.caption("Paste a news headline and description to get an instant trading signal.")

    col1, col2 = st.columns([2, 1])
    with col1:
        title = st.text_input("📰 News Headline", placeholder="e.g., Apple Reports Record Q4 Earnings")
        text = st.text_area(
            "📝 Article Description / Body",
            height=150,
            placeholder="Paste the news article content here...",
        )
    with col2:
        source = st.text_input("🏢 Source (optional)", placeholder="e.g., Reuters, Bloomberg")
        st.markdown("")
        st.markdown("")
        analyze_btn = st.button("🚀 Analyze", type="primary", width='stretch')

    if analyze_btn and (title or text):
        with st.spinner("Analyzing article..."):
            # Make sure model is trained
            if not pipeline.trading_model.is_trained:
                pipeline.train_model()

            result = pipeline.analyze_single_article(
                title=title,
                text=text,
                source=source,
                beginner_mode=beginner_mode,
            )

        render_single_result(result)

    elif analyze_btn:
        st.warning("Please enter at least a headline or article text.")


def render_single_result(result: dict):
    """Render the analysis result for a single article."""
    st.markdown("---")

    signal_data = result["signal"]
    sentiment = result["sentiment"]
    hype = result["hype"]

    # Signal badge
    signal = signal_data["signal"]
    badge_class = f"signal-{signal.lower()}"
    st.markdown(
        f'<div class="{badge_class}">🎯 Trading Signal: {signal}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Confidence", f"{signal_data['confidence']:.0%}")
    with col2:
        st.metric("Sentiment", f"{sentiment['sentiment_score']:+.2f}")
    with col3:
        st.metric("Hype Score", f"{hype['hype_score']:.0%}")
    with col4:
        st.metric("Risk Level", signal_data["risk_level"])

    # Probabilities gauge
    st.markdown("### 📊 Signal Probabilities")
    probs = signal_data.get("probabilities", {})
    cols = st.columns(3)
    for i, (label, prob) in enumerate(probs.items()):
        with cols[i]:
            color = {"Buy": "#2ecc71", "Sell": "#e74c3c", "Hold": "#f39c12"}.get(label, "#888")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                title={"text": label},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                },
            ))
            fig.update_layout(template="plotly_dark", height=250, margin=dict(t=50, b=0))
            st.plotly_chart(fig, width='stretch')

    # Full explanation
    st.markdown("### 📝 Detailed Explanation")
    st.text(signal_data["explanation"])

    # Hype analysis
    if hype["is_hype"]:
        st.error("⚠️ HYPE / MISLEADING CONTENT DETECTED")
        for flag in hype.get("flags", []):
            st.markdown(f"  • {flag}")
        st.markdown(f"**Hype Explanation:** {hype.get('explanation', '')}")
    else:
        st.success("✅ Article passes credibility check")

    # Sentiment details expander
    with st.expander("🔬 Sentiment Analysis Details"):
        st.json(sentiment)

    with st.expander("🔬 Hype Detection Details"):
        st.json({k: v for k, v in hype.items() if k != "explanation"})


# ─── Entry Point ──────────────────────────────────────────────────
if __name__ == "__main__":
    main()
