"""
News Fetcher — dual-mode:
  1. NewsAPI.org  (if a valid API key is configured in .streamlit/secrets.toml)
  2. Google News RSS fallback (no key required, always works)
"""

import requests
import re
import streamlit as st
from datetime import datetime, timedelta
from urllib.parse import quote


# ── NewsAPI ──────────────────────────────────────────────────────────────
NEWSAPI_BASE = "https://newsapi.org/v2/everything"


def _get_api_key() -> str:
    """Read the NewsAPI key from Streamlit secrets (returns '' if missing/placeholder)."""
    try:
        key = st.secrets["newsapi"]["api_key"]
        if key and "YOUR_" not in key.upper():
            return key
    except (KeyError, FileNotFoundError):
        pass
    return ""


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_newsapi(query: str, max_results: int = 15) -> list[dict]:
    """Fetch articles from NewsAPI (cached 10 min)."""
    api_key = _get_api_key()
    if not api_key:
        return []

    from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": min(max_results, 100),
        "apiKey": api_key,
    }

    try:
        resp = requests.get(NEWSAPI_BASE, params=params, timeout=10)
        data = resp.json()

        if data.get("status") != "ok":
            return []

        articles = []
        for art in data.get("articles", [])[:max_results]:
            published, time_ago = "", ""
            pub_str = art.get("publishedAt", "")
            if pub_str:
                try:
                    pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                    published = pub_dt.strftime("%Y-%m-%d %H:%M")
                    delta = datetime.utcnow() - pub_dt.replace(tzinfo=None)
                    time_ago = _format_time_ago(delta)
                except ValueError:
                    pass

            title = art.get("title", "")
            source_name = art.get("source", {}).get("name", "")
            if source_name and title.endswith(f" - {source_name}"):
                title = title[: -(len(source_name) + 3)].strip()

            articles.append({
                "title": title,
                "source": source_name,
                "published": published,
                "url": art.get("url", ""),
                "time_ago": time_ago,
                "description": art.get("description", ""),
            })

        return articles
    except Exception:
        return []


# ── Google News RSS (fallback) ───────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_google_news(query: str, max_results: int = 15) -> list[dict]:
    """Fetch headlines from Google News RSS (cached 10 min, no key needed)."""
    try:
        import feedparser
    except ImportError:
        # feedparser not installed — skip fallback
        return []

    encoded_query = quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries:
            return []

        articles = []
        for entry in feed.entries[:max_results]:
            published, time_ago = "", ""
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub_dt = datetime(*entry.published_parsed[:6])
                published = pub_dt.strftime("%Y-%m-%d %H:%M")
                delta = datetime.utcnow() - pub_dt
                time_ago = _format_time_ago(delta)

            title = entry.title
            source = ""
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0].strip()
                source = parts[1].strip()

            title = re.sub(r"<[^>]+>", "", title)

            articles.append({
                "title": title,
                "source": source,
                "published": published,
                "url": entry.link,
                "time_ago": time_ago,
            })

        return articles
    except Exception:
        return []


# ── Helpers ──────────────────────────────────────────────────────────────

def _format_time_ago(delta) -> str:
    """Convert timedelta to human-readable string."""
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days == 1:
        return "1d ago"
    return f"{days}d ago"


# Company name lookup for popular tickers
TICKER_NAMES = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "GOOGL": "Alphabet Google",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "META": "Meta Facebook",
    "NFLX": "Netflix",
    "AMD": "AMD",
    "BABA": "Alibaba",
}


def fetch_news_for_ticker(ticker: str, max_results: int = 15) -> list[dict]:
    """
    Fetch news for a stock ticker.
    Tries NewsAPI first (if key is set), then falls back to Google News RSS.
    """
    company = TICKER_NAMES.get(ticker.upper(), "")
    query = f"{ticker} stock"
    if company:
        query = f"{ticker} {company} stock"

    # Try NewsAPI first
    articles = _fetch_newsapi(query, max_results)
    if articles:
        return articles

    # Fallback to Google News RSS
    return _fetch_google_news(query, max_results)
