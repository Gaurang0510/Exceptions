"""
News2Trade AI - News Data Fetcher
Fetches financial news via:
  1. NewsAPI (primary, with your API key)
  2. Web scraping from major financial news sites (secondary/supplementary)
  3. Built-in sample dataset (fallback)
"""
import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from config import NEWS_API_KEY, NEWS_API_BASE_URL, FINANCIAL_KEYWORDS

# ─── Common Headers for Scraping ─────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ═══════════════════════════════════════════════════════════════════
#  1. NewsAPI Fetcher (Primary)
# ═══════════════════════════════════════════════════════════════════

def fetch_news_from_api(query: str = "stock market", page_size: int = 30) -> pd.DataFrame:
    """
    Fetch financial news articles from NewsAPI.
    Returns a DataFrame with title, description, source, publishedAt, url.
    """
    if not NEWS_API_KEY or NEWS_API_KEY == "your_newsapi_key_here":
        print("[INFO] No valid NEWS_API_KEY found. Skipping API fetch.")
        return pd.DataFrame()

    url = f"{NEWS_API_BASE_URL}/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
        "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok" or not data.get("articles"):
            print("[WARN] API returned no articles.")
            return pd.DataFrame()

        articles = []
        for art in data["articles"]:
            title = art.get("title", "") or ""
            desc = art.get("description", "") or ""
            # Skip removed/empty articles
            if not title or title == "[Removed]":
                continue
            articles.append({
                "title": title,
                "description": desc,
                "content": art.get("content", "") or "",
                "source": art.get("source", {}).get("name", "Unknown"),
                "published_at": art.get("publishedAt", ""),
                "url": art.get("url", ""),
                "data_source": "NewsAPI",
            })

        df = pd.DataFrame(articles)
        if not df.empty:
            df["text"] = df["title"].fillna("") + ". " + df["description"].fillna("")
            df["fetched_at"] = datetime.now().isoformat()
            print(f"[INFO] NewsAPI returned {len(df)} articles.")
        return df

    except Exception as e:
        print(f"[ERROR] API fetch failed: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
#  2. Web Scrapers (Secondary / Supplementary)
# ═══════════════════════════════════════════════════════════════════

def _safe_request(url: str, timeout: int = 10) -> BeautifulSoup | None:
    """Make a safe HTTP request and return parsed BeautifulSoup, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"[WARN] Scrape failed for {url}: {e}")
        return None


def scrape_reuters() -> list[dict]:
    """Scrape latest headlines from Reuters business section."""
    articles = []
    soup = _safe_request("https://www.reuters.com/business/")
    if not soup:
        return articles

    try:
        # Reuters uses data-testid attributes for article links
        for item in soup.select("a[data-testid='Heading']")[:15]:
            title = item.get_text(strip=True)
            href = item.get("href", "")
            if title and href:
                url = urljoin("https://www.reuters.com", href)
                articles.append({
                    "title": title,
                    "description": "",
                    "content": "",
                    "source": "Reuters (Scraped)",
                    "published_at": datetime.now().isoformat(),
                    "url": url,
                    "data_source": "Web Scraping",
                })
    except Exception as e:
        print(f"[WARN] Reuters parse error: {e}")

    # Fallback: try generic heading tags
    if not articles:
        for h3 in soup.find_all(["h3", "h2"], limit=15):
            a_tag = h3.find("a")
            if a_tag:
                title = a_tag.get_text(strip=True)
                href = a_tag.get("href", "")
                if title and len(title) > 15:
                    url = urljoin("https://www.reuters.com", href)
                    articles.append({
                        "title": title,
                        "description": "",
                        "content": "",
                        "source": "Reuters (Scraped)",
                        "published_at": datetime.now().isoformat(),
                        "url": url,
                        "data_source": "Web Scraping",
                    })

    print(f"[INFO] Reuters scraper: {len(articles)} articles")
    return articles


def scrape_cnbc() -> list[dict]:
    """Scrape latest headlines from CNBC markets."""
    articles = []
    soup = _safe_request("https://www.cnbc.com/markets/")
    if not soup:
        return articles

    try:
        for card in soup.select("a.Card-title, a.LatestNews-headline")[:15]:
            title = card.get_text(strip=True)
            href = card.get("href", "")
            if title and len(title) > 10:
                url = urljoin("https://www.cnbc.com", href)
                articles.append({
                    "title": title,
                    "description": "",
                    "content": "",
                    "source": "CNBC (Scraped)",
                    "published_at": datetime.now().isoformat(),
                    "url": url,
                    "data_source": "Web Scraping",
                })
    except Exception as e:
        print(f"[WARN] CNBC parse error: {e}")

    # Fallback
    if not articles:
        for h3 in soup.find_all(["h3", "h2", "h1"], limit=15):
            a_tag = h3.find("a") or h3.find_parent("a")
            title = h3.get_text(strip=True)
            href = (a_tag.get("href", "") if a_tag else "")
            if title and len(title) > 15:
                url = urljoin("https://www.cnbc.com", href) if href else ""
                articles.append({
                    "title": title,
                    "description": "",
                    "content": "",
                    "source": "CNBC (Scraped)",
                    "published_at": datetime.now().isoformat(),
                    "url": url,
                    "data_source": "Web Scraping",
                })

    print(f"[INFO] CNBC scraper: {len(articles)} articles")
    return articles


def scrape_marketwatch() -> list[dict]:
    """Scrape latest headlines from MarketWatch."""
    articles = []
    soup = _safe_request("https://www.marketwatch.com/latest-news")
    if not soup:
        return articles

    try:
        for item in soup.select("h3.article__headline a, a.link")[:15]:
            title = item.get_text(strip=True)
            href = item.get("href", "")
            if title and len(title) > 15:
                url = urljoin("https://www.marketwatch.com", href)
                articles.append({
                    "title": title,
                    "description": "",
                    "content": "",
                    "source": "MarketWatch (Scraped)",
                    "published_at": datetime.now().isoformat(),
                    "url": url,
                    "data_source": "Web Scraping",
                })
    except Exception as e:
        print(f"[WARN] MarketWatch parse error: {e}")

    # Fallback
    if not articles:
        for h3 in soup.find_all(["h3", "h2"], limit=15):
            a_tag = h3.find("a")
            if a_tag:
                title = a_tag.get_text(strip=True)
                href = a_tag.get("href", "")
                if title and len(title) > 15:
                    url = urljoin("https://www.marketwatch.com", href)
                    articles.append({
                        "title": title,
                        "description": "",
                        "content": "",
                        "source": "MarketWatch (Scraped)",
                        "published_at": datetime.now().isoformat(),
                        "url": url,
                        "data_source": "Web Scraping",
                    })

    print(f"[INFO] MarketWatch scraper: {len(articles)} articles")
    return articles


def scrape_yahoo_finance() -> list[dict]:
    """Scrape latest headlines from Yahoo Finance."""
    articles = []
    soup = _safe_request("https://finance.yahoo.com/news/")
    if not soup:
        return articles

    try:
        for item in soup.select("h3 a, li.js-stream-content a")[:15]:
            title = item.get_text(strip=True)
            href = item.get("href", "")
            if title and len(title) > 15:
                url = urljoin("https://finance.yahoo.com", href)
                articles.append({
                    "title": title,
                    "description": "",
                    "content": "",
                    "source": "Yahoo Finance (Scraped)",
                    "published_at": datetime.now().isoformat(),
                    "url": url,
                    "data_source": "Web Scraping",
                })
    except Exception as e:
        print(f"[WARN] Yahoo Finance parse error: {e}")

    # Fallback
    if not articles:
        for h3 in soup.find_all(["h3", "h2"], limit=15):
            a_tag = h3.find("a") or h3.find_parent("a")
            title = h3.get_text(strip=True)
            href = (a_tag.get("href", "") if a_tag else "")
            if title and len(title) > 15:
                url = urljoin("https://finance.yahoo.com", href) if href else ""
                articles.append({
                    "title": title,
                    "description": "",
                    "content": "",
                    "source": "Yahoo Finance (Scraped)",
                    "published_at": datetime.now().isoformat(),
                    "url": url,
                    "data_source": "Web Scraping",
                })

    print(f"[INFO] Yahoo Finance scraper: {len(articles)} articles")
    return articles


def scrape_google_news_rss(query: str = "stock market") -> list[dict]:
    """Scrape Google News RSS feed for a given query (no API key needed)."""
    articles = []
    try:
        rss_url = f"https://news.google.com/rss/search?q={query}+when:7d&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(rss_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "xml")

        for item in soup.find_all("item")[:20]:
            title = item.find("title")
            link = item.find("link")
            pub_date = item.find("pubDate")
            source_tag = item.find("source")

            title_text = title.get_text(strip=True) if title else ""
            if not title_text:
                continue

            articles.append({
                "title": title_text,
                "description": "",
                "content": "",
                "source": (source_tag.get_text(strip=True) if source_tag else "Google News") + " (RSS)",
                "published_at": pub_date.get_text(strip=True) if pub_date else datetime.now().isoformat(),
                "url": link.get_text(strip=True) if link else "",
                "data_source": "Google News RSS",
            })
    except Exception as e:
        print(f"[WARN] Google News RSS failed: {e}")

    print(f"[INFO] Google News RSS: {len(articles)} articles")
    return articles


def scrape_article_body(url: str) -> str:
    """
    Attempt to scrape the full article body text from a URL.
    Used to enrich scraped headlines with actual content.
    """
    if not url or "example.com" in url:
        return ""

    soup = _safe_request(url, timeout=8)
    if not soup:
        return ""

    try:
        # Remove non-content elements
        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Try article tag first
        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
        else:
            # Fall back to all paragraphs
            paragraphs = soup.find_all("p")

        text = " ".join(p.get_text(strip=True) for p in paragraphs[:20])

        # Clean up
        text = re.sub(r"\s+", " ", text).strip()
        return text[:2000]  # Cap at 2000 chars

    except Exception as e:
        print(f"[WARN] Body scrape failed for {url}: {e}")
        return ""


def _matches_query(title: str, query: str) -> bool:
    """
    Check if a headline is relevant to the user's search query.
    Returns True if any keyword from the query appears in the title.
    """
    if not query or not title:
        return False
    title_lower = title.lower()
    # Split query into individual keywords, ignore very short/common words
    stop_words = {"the", "a", "an", "and", "or", "is", "in", "on", "of", "to", "for", "it", "at", "by"}
    keywords = [w.strip().lower() for w in re.split(r'[\s,;]+', query) if len(w.strip()) > 1]
    keywords = [w for w in keywords if w not in stop_words]
    if not keywords:
        return True  # No meaningful keywords → accept everything
    return any(kw in title_lower for kw in keywords)


def fetch_scraped_news(query: str = "stock market", enrich_bodies: bool = False) -> pd.DataFrame:
    """
    Scrape news from multiple financial sites.
    Filters results to match the user's search query.
    Optionally enriches headlines by scraping article bodies.
    """
    all_articles = []

    # Run all scrapers
    scrapers = [
        ("Google News RSS", lambda: scrape_google_news_rss(query)),
        ("Reuters", scrape_reuters),
        ("CNBC", scrape_cnbc),
        ("MarketWatch", scrape_marketwatch),
        ("Yahoo Finance", scrape_yahoo_finance),
    ]

    for name, scraper_fn in scrapers:
        try:
            results = scraper_fn()
            all_articles.extend(results)
        except Exception as e:
            print(f"[WARN] {name} scraper failed: {e}")

    if not all_articles:
        print("[INFO] No articles scraped from any source.")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)

    # Filter scraped articles by query relevance
    before_filter = len(df)
    df = df[df["title"].apply(lambda t: _matches_query(t, query))].copy()
    print(f"[INFO] Query filter: {before_filter} → {len(df)} articles match '{query}'")

    if df.empty:
        print("[INFO] No scraped articles matched the query.")
        return pd.DataFrame()

    # De-duplicate by title (fuzzy: lowercase + strip)
    df["_title_lower"] = df["title"].str.lower().str.strip()
    df = df.drop_duplicates(subset="_title_lower").drop(columns="_title_lower")

    # Optionally enrich with article bodies
    if enrich_bodies:
        print("[INFO] Enriching scraped headlines with article body text...")
        bodies = []
        for url in df["url"]:
            body = scrape_article_body(url)
            bodies.append(body)
        df["content"] = bodies
        df["description"] = df["content"].apply(lambda x: x[:300] if x else "")

    # Build text field
    df["text"] = df["title"].fillna("") + ". " + df["description"].fillna("")
    df["fetched_at"] = datetime.now().isoformat()

    print(f"[INFO] Total scraped (deduplicated): {len(df)} articles")
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
#  3. Sample Dataset (Fallback)
# ═══════════════════════════════════════════════════════════════════

def get_sample_news() -> pd.DataFrame:
    """
    Provide a rich sample dataset of financial news for demo / fallback use.
    Covers: positive, negative, neutral, and hype/fake scenarios.
    """
    sample_articles = [
        # ── Positive / Bullish ───────────────────────────────────────
        {
            "title": "Apple Reports Record Q4 Earnings, Beating Analyst Expectations",
            "description": "Apple Inc. posted revenue of $94.8 billion for Q4 2025, up 8% YoY, driven by strong iPhone and Services growth.",
            "source": "Reuters", "published_at": "2025-12-15T10:30:00Z",
            "url": "https://example.com/apple-earnings",
        },
        {
            "title": "Tesla Stock Surges 12% After Announcing New Gigafactory in India",
            "description": "Tesla shares rallied after CEO confirmed plans for a $5B manufacturing plant in Maharashtra, India.",
            "source": "Bloomberg", "published_at": "2025-12-14T08:00:00Z",
            "url": "https://example.com/tesla-india",
        },
        {
            "title": "Federal Reserve Signals Potential Rate Cut in Early 2026",
            "description": "Fed Chair hinted at easing monetary policy as inflation cools to 2.1%, boosting market optimism.",
            "source": "CNBC", "published_at": "2025-12-13T14:00:00Z",
            "url": "https://example.com/fed-rate-cut",
        },
        {
            "title": "Microsoft Azure Revenue Grows 29%, Cloud Dominance Continues",
            "description": "Microsoft's cloud division posted strong results with Azure revenue growing 29% year-over-year.",
            "source": "TechCrunch", "published_at": "2025-12-12T09:00:00Z",
            "url": "https://example.com/msft-azure",
        },
        {
            "title": "India's GDP Growth Accelerates to 7.2% in Q3, Beating Estimates",
            "description": "Strong domestic consumption and exports drove India's GDP growth above analyst forecasts.",
            "source": "Economic Times", "published_at": "2025-12-11T06:30:00Z",
            "url": "https://example.com/india-gdp",
        },
        {
            "title": "NVIDIA Unveils Next-Gen AI Chip, Stock Hits All-Time High",
            "description": "NVIDIA's new Blackwell Ultra GPU promises 3x performance gains, sending shares up 9% in pre-market.",
            "source": "The Verge", "published_at": "2025-12-10T11:00:00Z",
            "url": "https://example.com/nvidia-chip",
        },
        {
            "title": "Amazon Announces $15 Billion Buyback Program",
            "description": "Amazon's board approved a massive share repurchase plan, signaling confidence in future growth.",
            "source": "Wall Street Journal", "published_at": "2025-12-09T12:00:00Z",
            "url": "https://example.com/amzn-buyback",
        },

        # ── Negative / Bearish ───────────────────────────────────────
        {
            "title": "Global Markets Plunge as China's Economy Shows Signs of Deflation",
            "description": "Major indices fell 3-4% as China reported negative CPI and weakening manufacturing data.",
            "source": "Financial Times", "published_at": "2025-12-14T07:00:00Z",
            "url": "https://example.com/china-deflation",
        },
        {
            "title": "Crypto Market Loses $200 Billion in 24 Hours Amid Regulatory Crackdown",
            "description": "Bitcoin dropped below $30,000 as SEC announced new enforcement actions against major exchanges.",
            "source": "CoinDesk", "published_at": "2025-12-13T16:00:00Z",
            "url": "https://example.com/crypto-crash",
        },
        {
            "title": "Boeing Shares Drop 15% After FAA Grounds 737 MAX Fleet Again",
            "description": "A new safety concern forces the FAA to ground all 737 MAX aircraft, hammering Boeing stock.",
            "source": "AP News", "published_at": "2025-12-12T13:00:00Z",
            "url": "https://example.com/boeing-grounded",
        },
        {
            "title": "Oil Prices Crash to $55 as OPEC+ Fails to Agree on Production Cuts",
            "description": "Brent crude plummeted 12% after OPEC+ members could not reach consensus on output reductions.",
            "source": "Reuters", "published_at": "2025-12-11T10:00:00Z",
            "url": "https://example.com/oil-crash",
        },
        {
            "title": "Major US Bank Reports $2 Billion Loss on Commercial Real Estate",
            "description": "Regional bank stocks fell across the board as loan defaults in office real estate surged.",
            "source": "Bloomberg", "published_at": "2025-12-10T08:30:00Z",
            "url": "https://example.com/bank-loss",
        },
        {
            "title": "Unemployment Claims Rise to 18-Month High, Recession Fears Grow",
            "description": "Weekly jobless claims jumped to 285K, well above expectations, sparking recession concerns.",
            "source": "CNBC", "published_at": "2025-12-09T14:00:00Z",
            "url": "https://example.com/unemployment",
        },

        # ── Neutral / Mixed ──────────────────────────────────────────
        {
            "title": "S&P 500 Closes Flat as Investors Await Fed Decision",
            "description": "Markets traded sideways with low volume as traders positioned ahead of the Fed meeting.",
            "source": "MarketWatch", "published_at": "2025-12-14T20:00:00Z",
            "url": "https://example.com/sp500-flat",
        },
        {
            "title": "Google Restructures Cloud Division, No Layoffs Expected",
            "description": "Alphabet reorganizes its Google Cloud unit for better enterprise focus. Analysts see limited impact.",
            "source": "TechCrunch", "published_at": "2025-12-13T09:00:00Z",
            "url": "https://example.com/google-cloud",
        },
        {
            "title": "European Central Bank Holds Interest Rates Steady at 3.5%",
            "description": "The ECB maintained its benchmark rate, in line with market expectations, citing balanced risk outlook.",
            "source": "Financial Times", "published_at": "2025-12-12T15:00:00Z",
            "url": "https://example.com/ecb-rates",
        },
        {
            "title": "Walmart Reports In-Line Earnings, Maintains Full-Year Guidance",
            "description": "Walmart met analyst expectations with steady grocery and e-commerce sales. Stock unchanged after hours.",
            "source": "Reuters", "published_at": "2025-12-11T17:00:00Z",
            "url": "https://example.com/walmart-earnings",
        },

        # ── Hype / Misleading / Clickbait ────────────────────────────
        {
            "title": "GUARANTEED 100x Returns! This Secret Stock Will Make You a Millionaire!",
            "description": "Insider sources reveal a penny stock that is guaranteed to skyrocket 10,000%. Act now before it's too late!",
            "source": "CryptoMoonShots Blog", "published_at": "2025-12-14T22:00:00Z",
            "url": "https://example.com/hype-stock",
        },
        {
            "title": "They Don't Want You to Know About This Crypto – 1000x Guaranteed!",
            "description": "Secret cryptocurrency that insiders are buying. Get rich quick with zero risk. Last chance to invest!",
            "source": "Unknown Blog", "published_at": "2025-12-13T21:00:00Z",
            "url": "https://example.com/hype-crypto",
        },
        {
            "title": "URGENT: Market Crash Coming Tomorrow – Sell Everything NOW!",
            "description": "Sources say a massive crash is imminent. You must sell all your holdings immediately or lose everything!",
            "source": "FearMonger Daily", "published_at": "2025-12-12T23:00:00Z",
            "url": "https://example.com/fear-hype",
        },
        {
            "title": "This AI Stock Will Explode 500% – Billionaires Are Secretly Buying",
            "description": "An explosive AI company nobody knows about is about to triple in value. Exclusive insider tip – act now!",
            "source": "StockTipsForFree", "published_at": "2025-12-11T20:00:00Z",
            "url": "https://example.com/ai-hype",
        },
        {
            "title": "FREE MONEY: Government Giving $50,000 to Every Investor This Month",
            "description": "No risk required! The government is giving away free money to investors. Sign up now before slots run out!",
            "source": "ScamNewsDaily", "published_at": "2025-12-10T19:00:00Z",
            "url": "https://example.com/scam-news",
        },

        # ── More Realistic Positive ──────────────────────────────────
        {
            "title": "JPMorgan Raises S&P 500 Year-End Target to 5,500",
            "description": "JPMorgan strategists upgraded their market outlook citing resilient consumer spending and AI-driven productivity.",
            "source": "Bloomberg", "published_at": "2025-12-09T07:00:00Z",
            "url": "https://example.com/jpmorgan-target",
        },
        {
            "title": "Renewable Energy Stocks Rally as New Climate Bill Passes Senate",
            "description": "Solar and wind energy companies surged 8-15% after bipartisan climate legislation cleared the Senate.",
            "source": "CNBC", "published_at": "2025-12-08T11:00:00Z",
            "url": "https://example.com/renewable-rally",
        },
        {
            "title": "Pharmaceutical Giant Pfizer Gets FDA Approval for New Cancer Drug",
            "description": "Pfizer stock jumped 6% after FDA granted approval for its novel immunotherapy treatment for lung cancer.",
            "source": "Reuters", "published_at": "2025-12-07T09:00:00Z",
            "url": "https://example.com/pfizer-fda",
        },

        # ── More Realistic Negative ──────────────────────────────────
        {
            "title": "Meta Faces Record $3 Billion EU Antitrust Fine",
            "description": "European Commission levied its largest ever tech penalty on Meta for anti-competitive practices.",
            "source": "Financial Times", "published_at": "2025-12-08T13:00:00Z",
            "url": "https://example.com/meta-fine",
        },
        {
            "title": "Supply Chain Disruptions Return as Red Sea Shipping Routes Close",
            "description": "Global shipping costs surge 40% as major carriers reroute around Africa, threatening holiday retail season.",
            "source": "Wall Street Journal", "published_at": "2025-12-07T08:00:00Z",
            "url": "https://example.com/shipping-crisis",
        },
        {
            "title": "Intel Announces 10,000 Layoffs as PC Demand Continues to Slump",
            "description": "Intel will cut 10% of its workforce as the company struggles with declining PC sales and market share loss.",
            "source": "The Verge", "published_at": "2025-12-06T10:00:00Z",
            "url": "https://example.com/intel-layoffs",
        },
    ]

    df = pd.DataFrame(sample_articles)
    df["content"] = df["description"]
    df["text"] = df["title"].fillna("") + ". " + df["description"].fillna("")
    df["fetched_at"] = datetime.now().isoformat()
    df["data_source"] = "Sample Dataset"
    return df


# ═══════════════════════════════════════════════════════════════════
#  4. Unified Fetch Function
# ═══════════════════════════════════════════════════════════════════

def fetch_news(
    query: str = "stock market finance",
    page_size: int = 30,
    use_api: bool = True,
    use_scraping: bool = True,
    enrich_bodies: bool = False,
) -> pd.DataFrame:
    """
    Main entry point — combines all data sources:
      1. NewsAPI (if key available)
      2. Web scraping (Reuters, CNBC, MarketWatch, Yahoo Finance, Google News RSS)
      3. Sample dataset (fallback if nothing else works)

    Args:
        query: Search query for news
        page_size: Maximum number of articles to return (hard cap)
        use_api: Whether to use NewsAPI
        use_scraping: Whether to scrape websites
        enrich_bodies: Whether to scrape full article bodies (slower)

    Returns:
        Combined, deduplicated, query-filtered DataFrame capped at page_size.
    """
    frames = []

    # 1. NewsAPI — over-fetch so we have enough after relevance filtering
    if use_api:
        api_df = fetch_news_from_api(query, min(page_size * 3, 100))
        if not api_df.empty:
            frames.append(api_df)

    # 2. Web Scraping
    if use_scraping:
        scraped_df = fetch_scraped_news(query, enrich_bodies=enrich_bodies)
        if not scraped_df.empty:
            frames.append(scraped_df)

    # 3. Combine
    if frames:
        combined = pd.concat(frames, ignore_index=True)

        # Deduplicate by lowercase title
        combined["_title_lower"] = combined["title"].str.lower().str.strip()
        combined = combined.drop_duplicates(subset="_title_lower").drop(columns="_title_lower")

        # Ensure required columns exist
        for col in ["text", "content", "description", "data_source"]:
            if col not in combined.columns:
                combined[col] = ""

        # Re-build text if needed
        mask = combined["text"].isna() | (combined["text"].str.strip() == "")
        combined.loc[mask, "text"] = (
            combined.loc[mask, "title"].fillna("") + ". " +
            combined.loc[mask, "description"].fillna("")
        )

        # Filter ALL articles by query relevance (catches irrelevant API results too)
        before = len(combined)
        combined = combined[combined["title"].apply(lambda t: _matches_query(t, query))].copy()
        if len(combined) < before:
            print(f"[INFO] Relevance filter: {before} → {len(combined)} articles match '{query}'")

        # Enforce page_size as a hard cap on total articles
        if len(combined) > page_size:
            combined = combined.head(page_size)
            print(f"[INFO] Capped results to {page_size} articles (user-requested limit).")

        print(f"[INFO] Total articles (combined): {len(combined)}")
        return combined.reset_index(drop=True)

    # 4. Fallback to sample data
    print("[INFO] No live data available. Using sample dataset.")
    return get_sample_news()
