"""
Financial Sentiment Analyzer using VADER
Lightweight (~1 MB) rule-based sentiment scoring tuned for social media / news text.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Module-level singleton so the model is loaded once
_analyzer = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def analyze_text(text: str) -> dict:
    """
    Score a single piece of text.

    Returns:
        dict with keys:
            score   – float in [-1, +1]  (compound score)
            label   – 'Bullish' | 'Bearish' | 'Neutral'
            pos, neg, neu – component scores
    """
    analyzer = _get_analyzer()
    scores = analyzer.polarity_scores(text)

    compound = scores['compound']
    if compound >= 0.15:
        label = 'Bullish'
    elif compound <= -0.15:
        label = 'Bearish'
    else:
        label = 'Neutral'

    return {
        'score': compound,
        'label': label,
        'pos': scores['pos'],
        'neg': scores['neg'],
        'neu': scores['neu'],
    }


def analyze_batch(texts: list[str]) -> list[dict]:
    """Score a list of texts and return list of result dicts."""
    return [analyze_text(t) for t in texts]


def get_aggregate_sentiment(texts: list[str]) -> dict:
    """
    Aggregate sentiment across multiple headlines / texts.

    Returns:
        dict with keys:
            overall_score – mean compound score  [-1, +1]
            label         – Bullish / Bearish / Neutral
            bullish_pct   – % of texts that are bullish
            bearish_pct   – % of texts that are bearish
            neutral_pct   – % of texts that are neutral
            count         – total number of texts scored
            items         – list of individual results
    """
    if not texts:
        return {
            'overall_score': 0.0,
            'label': 'Neutral',
            'bullish_pct': 0,
            'bearish_pct': 0,
            'neutral_pct': 100,
            'count': 0,
            'items': [],
        }

    results = analyze_batch(texts)

    scores = [r['score'] for r in results]
    overall = sum(scores) / len(scores)

    bullish = sum(1 for r in results if r['label'] == 'Bullish')
    bearish = sum(1 for r in results if r['label'] == 'Bearish')
    neutral = sum(1 for r in results if r['label'] == 'Neutral')
    total = len(results)

    if overall >= 0.15:
        label = 'Bullish'
    elif overall <= -0.15:
        label = 'Bearish'
    else:
        label = 'Neutral'

    return {
        'overall_score': round(overall, 4),
        'label': label,
        'bullish_pct': round(bullish / total * 100, 1),
        'bearish_pct': round(bearish / total * 100, 1),
        'neutral_pct': round(neutral / total * 100, 1),
        'count': total,
        'items': results,
    }
