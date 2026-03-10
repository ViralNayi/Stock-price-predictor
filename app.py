"""
Stock Price Prediction with LSTM — Intelligence Dashboard
Main Streamlit Application
"""

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import html as html_mod
from datetime import datetime, timedelta, date
import plotly.graph_objects as go

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.data_processor import prepare_data_pipeline
from model.lstm_model import build_model, create_early_stopping, save_model, load_model
from model.trainer import train_model, make_predictions, evaluate_model, inverse_transform_close, predict_future
from utils.visualizations import plot_stock_history, plot_predictions, plot_training_history, display_metrics

# ── Page Configuration ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Intelligence Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Custom CSS ──────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "styles", "dashboard.css")
if os.path.exists(css_path):
    with open(css_path, 'r') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Sidebar Configuration ───────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ CONFIGURATION")

POPULAR_TICKERS = [
    "AAPL – Apple",
    "TSLA – Tesla",
    "GOOGL – Alphabet",
    "MSFT – Microsoft",
    "NVDA – NVIDIA",
    "AMZN – Amazon",
    "META – Meta",
    "NFLX – Netflix",
    "AMD – AMD",
    "BABA – Alibaba",
    "✏️ Type custom ticker…",
]

selected = st.sidebar.selectbox(
    "Stock Ticker",
    options=POPULAR_TICKERS,
    index=0,
    help="Pick a popular ticker or choose the last option to type your own"
)

if selected.startswith("✏️"):
    ticker = st.sidebar.text_input(
        "Enter ticker symbol",
        value="",
        placeholder="e.g. RELIANCE.NS, BRK-B, SPY",
        help="Type any valid Yahoo Finance ticker symbol"
    ).strip().upper()
    if not ticker:
        st.sidebar.warning("⚠️ Please enter a ticker symbol above.")
else:
    ticker = selected.split(" – ")[0].strip()

# Date range
st.sidebar.markdown("### 📅 DATE RANGE")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start",
        value=datetime.now() - timedelta(days=365 * 5),
        max_value=datetime.now()
    )
with col2:
    end_date = st.date_input(
        "End",
        value=datetime.now(),
        max_value=datetime.now()
    )

# Model parameters
st.sidebar.markdown("### 🔧 MODEL PARAMS")

seq_length = st.sidebar.slider("Sequence Length", 30, 120, 60, 10,
                                help="Past days used for prediction")
epochs = st.sidebar.slider("Epochs", 10, 100, 50, 10,
                            help="Training iterations")
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

# Forecast
st.sidebar.markdown("### 🔮 FORECAST")
forecast_days = st.sidebar.slider("Forecast Days", 1, 90, 30, 1,
                                   help="Future trading days to predict")

# Train button
train_button = st.sidebar.button("🚀 TRAIN & PREDICT", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
#  HELPER: Fetch live price snippet from yfinance
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def get_live_price(tkr: str):
    """Return latest price info for the header bar."""
    import yfinance as yf
    try:
        t = yf.Ticker(tkr)
        info = t.fast_info
        price = info.get('lastPrice', info.get('last_price', 0))
        prev_close = info.get('previousClose', info.get('previous_close', price))
        change = price - prev_close
        pct = (change / prev_close * 100) if prev_close else 0
        return {'price': price, 'change': change, 'pct': pct}
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_technical_indicators(tkr: str):
    """Fetch 3-month history and compute RSI, MACD, SMA, EMA (cached 5 min)."""
    import yfinance as yf
    try:
        t = yf.Ticker(tkr)
        hist = t.history(period="3mo")
        if hist.empty:
            return None
        close = hist['Close']
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - 100 / (1 + rs)).iloc[-1]
        # MACD
        ema12 = close.ewm(span=12).mean().iloc[-1]
        ema26 = close.ewm(span=26).mean().iloc[-1]
        macd_val = ema12 - ema26
        # SMA
        sma20 = close.rolling(20).mean().iloc[-1]
        return {'rsi': rsi, 'macd': macd_val, 'sma20': sma20, 'ema12': ema12}
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
#  HELPER: Render HTML components
# ═══════════════════════════════════════════════════════════════════════
def render_header_bar(tkr, price_info):
    """Render the top header bar with ticker + live price."""
    if price_info and price_info['price']:
        p = price_info
        change_class = 'price-change-up' if p['change'] >= 0 else 'price-change-down'
        arrow = '▲' if p['change'] >= 0 else '▼'
        st.markdown(f"""
        <div class="header-bar">
            <span class="ticker-name">{tkr}</span>
            <span class="live-price">${p['price']:.2f}</span>
            <span class="{change_class}">{arrow} {abs(p['change']):.2f} ({abs(p['pct']):.2f}%)</span>
            <span style="margin-left:auto; color:#6b7280; font-family:var(--font-mono); font-size:0.75rem;">
                <span class="status-dot live"></span>LIVE
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="header-bar">
            <span class="ticker-name">{tkr}</span>
            <span style="color:#6b7280; font-family:var(--font-mono); font-size:0.85rem;">
                Select ticker & train model to see data
            </span>
        </div>
        """, unsafe_allow_html=True)


def render_sentiment_panel(sentiment_data, news_items):
    """Render the AI Sentiment panel."""
    import streamlit.components.v1 as components

    s = sentiment_data
    score = s['overall_score']

    if s['label'] == 'Bullish':
        score_color, bar_color = '#10b981', '#10b981'
    elif s['label'] == 'Bearish':
        score_color, bar_color = '#ef4444', '#ef4444'
    else:
        score_color, bar_color = '#f59e0b', '#f59e0b'

    bar_pct = int((score + 1) / 2 * 100)

    # Contextual hint based on score
    if score >= 0.5:
        hint = "📈 Strong positive sentiment — market outlook is optimistic"
    elif score >= 0.15:
        hint = "📈 Mildly positive — news leans bullish"
    elif score <= -0.5:
        hint = "📉 Strong negative sentiment — market outlook is pessimistic"
    elif score <= -0.15:
        hint = "📉 Mildly negative — news leans bearish"
    else:
        hint = "➖ Mixed signals — no clear direction from news"

    panel_html = f"""
    <div style="background:#111827;border:1px solid #1e2a3a;border-radius:8px;padding:16px 20px;">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #1e2a3a;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;font-weight:600;color:#9ca3af;text-transform:uppercase;letter-spacing:2px;">🤖 AI Sentiment</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#6b7280;">
                <span style="width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px;background:{bar_color};box-shadow:0 0 6px {bar_color};animation:pulse 2s infinite;"></span>VADER
            </span>
        </div>
        <div style="text-align:center;padding:12px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:2.5rem;font-weight:700;line-height:1;color:{score_color};">{score:+.2f}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:2px;margin-top:4px;color:{bar_color};">{s['label']}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#9ca3af;margin-top:8px;line-height:1.4;">{hint}</div>
            <div style="width:100%;height:6px;background:#0a0e17;border-radius:3px;margin-top:12px;overflow:hidden;">
                <div style="height:100%;border-radius:3px;width:{bar_pct}%;background:{bar_color};transition:width 0.5s ease;"></div>
            </div>
        </div>
        <div style="margin-top:14px;font-family:'JetBrains Mono',monospace;font-size:0.7rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">
                <span style="color:#10b981;">▲ {s['bullish_pct']}%</span>
                <span style="color:#6b7280;font-size:0.6rem;">Positive headlines</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">
                <span style="color:#f59e0b;">● {s['neutral_pct']}%</span>
                <span style="color:#6b7280;font-size:0.6rem;">No clear signal</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">
                <span style="color:#ef4444;">▼ {s['bearish_pct']}%</span>
                <span style="color:#6b7280;font-size:0.6rem;">Negative headlines</span>
            </div>
        </div>
        <div style="margin-top:10px;padding-top:8px;border-top:1px solid #1e2a3a;font-family:'JetBrains Mono',monospace;font-size:0.55rem;color:#4b5563;text-align:center;">
            Score range: −1.0 (very bearish) to +1.0 (very bullish)
        </div>
    </div>
    <style>@keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }} }}</style>
    """
    components.html(panel_html, height=330, scrolling=False)


def render_news_feed(news_items, sentiment_results):
    """Render the live news feed with sentiment tags."""
    import streamlit.components.v1 as components

    items_html = ""
    for i, article in enumerate(news_items[:10]):
        if i < len(sentiment_results):
            result = sentiment_results[i]
            tag_class = result['label'].lower()
            tag_text = result['label'].upper()
        else:
            tag_class = 'neutral'
            tag_text = 'NEUTRAL'

        # HTML-escape dynamic content to prevent broken rendering
        safe_title = html_mod.escape(article['title'])
        safe_source = html_mod.escape(article.get('source', ''))
        source_text = f' &middot; {safe_source}' if safe_source else ''
        time_text = html_mod.escape(article.get('time_ago', ''))

        items_html += f"""
        <div style="padding:10px 0;border-bottom:1px solid #1e2a3a;display:flex;align-items:flex-start;gap:10px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;font-weight:600;padding:2px 8px;border-radius:3px;text-transform:uppercase;letter-spacing:1px;white-space:nowrap;flex-shrink:0;
                {'background:#10b98120;color:#10b981;border:1px solid #10b98140;' if tag_class == 'bullish' else
                 'background:#ef444420;color:#ef4444;border:1px solid #ef444440;' if tag_class == 'bearish' else
                 'background:#f59e0b20;color:#f59e0b;border:1px solid #f59e0b40;'}">{tag_text}</span>
            <div>
                <div style="font-size:0.85rem;color:#e5e7eb;line-height:1.3;">{safe_title}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#6b7280;margin-top:3px;">{time_text}{source_text}</div>
            </div>
        </div>
        """

    full_html = f"""
    <div style="background:#111827;border:1px solid #1e2a3a;border-radius:8px;padding:16px 20px;">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #1e2a3a;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;font-weight:600;color:#9ca3af;text-transform:uppercase;letter-spacing:2px;">📰 Live News Feed</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#6b7280;">{len(news_items)} articles</span>
        </div>
        <div style="max-height:350px;overflow-y:auto;">
            {items_html if items_html else '<p style="color:#6b7280;font-size:0.85rem;">No news found for this ticker.</p>'}
        </div>
    </div>
    """
    components.html(full_html, height=420, scrolling=False)


def render_indicator_card(label, value, color="#e5e7eb", hint="", hint_color="#6b7280"):
    """Render a small technical indicator card with an optional hint."""
    hint_html = f'<div style="font-family:var(--font-mono);font-size:0.65rem;color:{hint_color};margin-top:4px;letter-spacing:0.5px;">{hint}</div>' if hint else ''
    st.markdown(f"""
    <div class="indicator-card">
        <div class="indicator-label">{label}</div>
        <div class="indicator-value" style="color:{color};">{value}</div>
        {hint_html}
    </div>
    """, unsafe_allow_html=True)


def render_forecast_banner(next_day_price, next_day_date, last_price):
    """Render the next-day forecast banner."""
    change = next_day_price - last_price
    pct = (change / last_price) * 100
    if change >= 0:
        arrow = '▲'
        color = '#10b981'
        bg = '#10b98112'
    else:
        arrow = '▼'
        color = '#ef4444'
        bg = '#ef444412'

    st.markdown(f"""
    <div class="forecast-banner">
        <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
            <div>
                <div style="font-family:var(--font-mono); font-size:0.7rem; color:#6b7280; text-transform:uppercase; letter-spacing:1.5px;">
                    Next Trading Day
                </div>
                <div style="font-family:var(--font-mono); font-size:0.75rem; color:#9ca3af; margin-top:2px;">
                    {next_day_date}
                </div>
            </div>
            <div style="font-family:var(--font-mono); font-size:1.5rem; font-weight:700; color:#e5e7eb;">
                ${next_day_price:.2f}
            </div>
            <div style="font-family:var(--font-mono); font-size:1rem; font-weight:600; color:{color}; background:{bg}; padding:4px 12px; border-radius:4px;">
                {arrow} ${abs(change):.2f} ({abs(pct):.2f}%)
            </div>
            <div style="font-family:var(--font-mono); font-size:0.75rem; color:#6b7280; margin-left:auto;">
                vs last close <span style="color:#e5e7eb;">${last_price:.2f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

# ── Header Bar ────────────────────────────────────────────────────────
if ticker:
    price_info = get_live_price(ticker)
    render_header_bar(ticker, price_info)
else:
    render_header_bar("—", None)

# ── Sentiment & News (always visible, even before training) ──────────
if ticker:
    from sentiment.news_fetcher import fetch_news_for_ticker
    from sentiment.analyzer import analyze_batch, get_aggregate_sentiment

    news_items = fetch_news_for_ticker(ticker, max_results=15)
    headlines = [a['title'] for a in news_items]

    if headlines:
        sentiment_results = analyze_batch(headlines)
        aggregate = get_aggregate_sentiment(headlines)
    else:
        sentiment_results = []
        aggregate = {
            'overall_score': 0.0, 'label': 'Neutral',
            'bullish_pct': 0, 'bearish_pct': 0, 'neutral_pct': 100,
            'count': 0, 'items': [],
        }

    # Layout: sentiment gauge + news feed side by side
    sent_col, news_col = st.columns([1, 2])

    with sent_col:
        render_sentiment_panel(aggregate, news_items)

        # Technical indicators preview (from latest data)
        st.markdown("""
        <div class="dash-panel">
            <div class="dash-panel-header">
                <span class="dash-panel-title">📊 Market Indicators</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if ticker and price_info and price_info['price']:
            indicators = get_technical_indicators(ticker)
            if indicators:
                rsi = indicators['rsi']
                macd_val = indicators['macd']
                sma20 = indicators['sma20']
                ema12 = indicators['ema12']

                rsi_color = '#10b981' if 30 < rsi < 70 else '#ef4444'
                if rsi >= 70:
                    rsi_hint, rsi_hint_color = '⚠ Overbought — may drop', '#ef4444'
                elif rsi <= 30:
                    rsi_hint, rsi_hint_color = '⚠ Oversold — may bounce', '#ef4444'
                elif rsi > 55:
                    rsi_hint, rsi_hint_color = '↑ Leaning bullish', '#10b981'
                elif rsi < 45:
                    rsi_hint, rsi_hint_color = '↓ Leaning bearish', '#f59e0b'
                else:
                    rsi_hint, rsi_hint_color = '● Neutral zone', '#6b7280'
                render_indicator_card("RSI (14)", f"{rsi:.1f}", rsi_color, rsi_hint, rsi_hint_color)

                macd_color = '#10b981' if macd_val >= 0 else '#ef4444'
                if macd_val > 0:
                    macd_hint, macd_hint_color = '▲ Bullish momentum', '#10b981'
                elif macd_val < 0:
                    macd_hint, macd_hint_color = '▼ Bearish momentum', '#ef4444'
                else:
                    macd_hint, macd_hint_color = '● Crossover point', '#f59e0b'
                render_indicator_card("MACD", f"{macd_val:.2f}", macd_color, macd_hint, macd_hint_color)

                last_close = price_info['price']
                if last_close > sma20:
                    sma_hint, sma_hint_color = '▲ Price above support', '#10b981'
                else:
                    sma_hint, sma_hint_color = '▼ Price below support', '#ef4444'
                render_indicator_card("SMA 20", f"${sma20:.2f}", hint=sma_hint, hint_color=sma_hint_color)

                if last_close > ema12:
                    ema_hint, ema_hint_color = '▲ Short-term uptrend', '#10b981'
                else:
                    ema_hint, ema_hint_color = '▼ Short-term downtrend', '#ef4444'
                render_indicator_card("EMA 12", f"${ema12:.2f}", hint=ema_hint, hint_color=ema_hint_color)

    with news_col:
        render_news_feed(news_items, sentiment_results)


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING & PREDICTION SECTION
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")

if train_button:
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Fetch and prepare data
        status_text.text("📊 Fetching stock data & computing indicators...")
        progress_bar.progress(10)

        data_dict = prepare_data_pipeline(
            ticker=ticker,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            seq_length=seq_length,
            train_ratio=0.8
        )

        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        scaler = data_dict['scaler']
        original_df = data_dict['original_df']
        n_features = data_dict['n_features']
        feature_df = data_dict['feature_df']

        st.success(f"✅ Loaded {len(original_df)} days  ·  {n_features} features  ·  Train/Test split at 80%")

        # Step 2: Historical chart
        st.markdown("""
        <div class="dash-panel">
            <div class="dash-panel-header">
                <span class="dash-panel-title">📈 Historical Price Data</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        fig_history = plot_stock_history(original_df, ticker)
        st.plotly_chart(fig_history, use_container_width=True)

        # Step 3: Build model
        status_text.text("🏗️ Building LSTM model...")
        progress_bar.progress(30)
        model = build_model(input_shape=(seq_length, n_features))

        # Step 4: Train model
        status_text.text("🎓 Training model... This may take a few minutes")
        progress_bar.progress(40)
        early_stop = create_early_stopping(patience=10)
        history = train_model(
            model=model,
            X_train=X_train, y_train=y_train,
            X_val=X_test, y_val=y_test,
            epochs=epochs, batch_size=batch_size,
            callbacks=[early_stop]
        )
        progress_bar.progress(70)

        # Step 5: Predictions
        status_text.text("🔮 Making predictions...")
        y_pred_normalized = make_predictions(model, X_test)
        y_test_original = inverse_transform_close(y_test, scaler, n_features)
        y_pred_original = inverse_transform_close(y_pred_normalized, scaler, n_features)
        progress_bar.progress(85)

        # Step 6: Evaluate
        status_text.text("📊 Evaluating performance...")
        metrics = evaluate_model(y_test_original, y_pred_original)
        progress_bar.progress(100)
        status_text.text("✅ Training complete!")

        # ── Metrics Panel ─────────────────────────────────────────────
        st.markdown("""
        <div class="dash-panel">
            <div class="dash-panel-header">
                <span class="dash-panel-title">🎯 Model Performance</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        display_metrics(metrics)

        # ── Predictions Chart ─────────────────────────────────────────
        st.markdown("""
        <div class="dash-panel">
            <div class="dash-panel-header">
                <span class="dash-panel-title">📈 Predictions vs Actual</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        train_size = len(X_train)
        test_dates = original_df.index[seq_length + train_size:]
        fig_pred = plot_predictions(
            dates=test_dates,
            actual=y_test_original.flatten(),
            predicted=y_pred_original.flatten(),
            train_size=0
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # ── Future Forecast ───────────────────────────────────────────
        st.markdown(f"""
        <div class="dash-panel">
            <div class="dash-panel-header">
                <span class="dash-panel-title">🔮 Future Forecast — {forecast_days} Trading Days</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        normalized_full = data_dict['normalized_data']
        future_prices = predict_future(
            model=model,
            normalized_data=normalized_full,
            scaler=scaler,
            feature_df=feature_df,
            seq_length=seq_length,
            n_days=forecast_days
        )

        # Build future date index
        today = pd.Timestamp(date.today())
        last_date = original_df.index[-1]
        last_date_naive = last_date.tz_localize(None) if last_date.tzinfo is not None else last_date
        anchor = max(last_date_naive.normalize(), today)
        future_dates = pd.bdate_range(start=anchor + pd.Timedelta(days=1), periods=forecast_days)

        # Chart: historical close + forecast
        fig_future = go.Figure()
        history_window = min(90, len(original_df))
        hist_dates = original_df.index[-history_window:]
        hist_prices = original_df['Close'].values[-history_window:]

        fig_future.add_trace(go.Scatter(
            x=hist_dates, y=hist_prices,
            mode='lines', name='Historical Close',
            line=dict(color='#10b981', width=2)
        ))
        fig_future.add_trace(go.Scatter(
            x=future_dates, y=future_prices.flatten(),
            mode='lines+markers', name='Forecast',
            line=dict(color='#f59e0b', width=2, dash='dot'),
            marker=dict(size=5)
        ))
        fig_future.add_vrect(
            x0=str(future_dates[0].date()),
            x1=str(future_dates[-1].date()),
            fillcolor='rgba(245,158,11,0.07)',
            layer='below', line_width=0
        )
        fig_future.update_layout(
            title=f'{ticker} — {forecast_days}-Day Price Forecast',
            xaxis_title='Date', yaxis_title='Price (USD)',
            xaxis=dict(constrain='domain'),
            yaxis=dict(constrain='domain'),
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='#0a0e17',
            plot_bgcolor='#0a0e17',
            height=450,
            legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
            font=dict(family='JetBrains Mono, monospace')
        )
        st.plotly_chart(fig_future, use_container_width=True)

        # ── Next-Day Forecast Banner ──────────────────────────────────
        next_day_price = future_prices[0, 0]
        next_day_date = future_dates[0].strftime('%A, %b %d %Y')
        last_price = float(original_df['Close'].iloc[-1])
        render_forecast_banner(next_day_price, next_day_date, last_price)

        # ── Download buttons row ──────────────────────────────────────
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            forecast_df = pd.DataFrame({
                'Date': future_dates.strftime('%Y-%m-%d'),
                'Forecast Price': future_prices.flatten()
            })
            st.download_button(
                label="📥 Download Forecast CSV",
                data=forecast_df.to_csv(index=False),
                file_name=f"{ticker}_forecast_{forecast_days}days.csv",
                mime="text/csv"
            )
        with dl_col2:
            results_df = pd.DataFrame({
                'Date': test_dates,
                'Actual Price': y_test_original.flatten(),
                'Predicted Price': y_pred_original.flatten(),
                'Error': y_test_original.flatten() - y_pred_original.flatten()
            })
            st.download_button(
                label="📥 Download Predictions CSV",
                data=results_df.to_csv(index=False),
                file_name=f"{ticker}_predictions.csv",
                mime="text/csv"
            )

        # ── Training Progress Chart ───────────────────────────────────
        with st.expander("📉 Training Progress", expanded=False):
            fig_history_plot = plot_training_history(history)
            st.plotly_chart(fig_history_plot, use_container_width=True)

        # ── Save Model ────────────────────────────────────────────────
        with st.expander("💾 Save Model", expanded=False):
            if st.button("Save Trained Model"):
                model_path = f"saved_models/{ticker}_lstm_model.h5"
                save_model(model, model_path)
                st.success(f"Model saved to {model_path}")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    # ── Initial State — Welcome Panel ─────────────────────────────────
    st.markdown("""
    <div class="dash-panel" style="text-align:center; padding:40px;">
        <div style="font-family:var(--font-mono); font-size:0.8rem; color:#6b7280; text-transform:uppercase; letter-spacing:3px; margin-bottom:12px;">
            LSTM Intelligence System
        </div>
        <div style="font-family:var(--font-mono); font-size:1.2rem; color:#e5e7eb; margin-bottom:8px;">
            Configure parameters in the sidebar
        </div>
        <div style="font-family:var(--font-mono); font-size:0.85rem; color:#6b7280;">
            Click <span style="color:#f59e0b;">TRAIN & PREDICT</span> to start analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature info
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("""
        <div class="dash-panel">
            <div class="dash-panel-header">
                <span class="dash-panel-title">🧠 Model Architecture</span>
            </div>
            <div style="font-size:0.85rem; line-height:1.8;">
                • 3 LSTM layers × 50 units<br>
                • 7 input features: Close, Volume, RSI, MACD, Signal, SMA, EMA<br>
                • Dropout regularization (20%)<br>
                • Adam optimizer + MSE loss<br>
                • Early stopping with patience=10
            </div>
        </div>
        """, unsafe_allow_html=True)

    with info_col2:
        st.markdown("""
        <div class="dash-panel">
            <div class="dash-panel-header">
                <span class="dash-panel-title">📡 Sentiment Engine</span>
            </div>
            <div style="font-size:0.85rem; line-height:1.8;">
                • VADER sentiment scoring<br>
                • NewsAPI + Google News RSS fallback<br>
                • Real-time headline analysis<br>
                • Bullish / Bearish / Neutral classification<br>
                • Aggregate sentiment gauge
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  COMPARE STOCKS SECTION
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
COMPARE_COLORS = ['#10b981', '#3b82f6', '#f59e0b']

with st.expander("📊 Compare Stocks — Train & Compare 2-3 Tickers", expanded=False):
    st.markdown("""
    <div style="font-family:var(--font-mono); font-size:0.85rem; color:#9ca3af; margin-bottom:12px;">
        Select 2-3 tickers to compare sentiment, model performance, and forecasts side by side.
        Uses reduced epochs (25) and 14-day forecast for faster comparison.
    </div>
    """, unsafe_allow_html=True)

    compare_tickers = st.multiselect(
        "Select tickers to compare",
        options=["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "AMZN", "META", "NFLX", "AMD", "BABA"],
        default=["AAPL", "TSLA"],
        max_selections=3,
        help="Pick 2-3 tickers for comparison"
    )

    compare_button = st.button("⚡ COMPARE", type="primary", use_container_width=True, key="compare_btn")

    if len(compare_tickers) < 2:
        st.warning("⚠️ Please select at least 2 tickers to compare.")

    elif compare_button:
        try:
            from sentiment.news_fetcher import fetch_news_for_ticker as cmp_fetch_news
            from sentiment.analyzer import get_aggregate_sentiment as cmp_get_agg

            n_tickers = len(compare_tickers)

            # ── Sentiment Comparison ──────────────────────────────────
            st.markdown("""
            <div class="dash-panel">
                <div class="dash-panel-header">
                    <span class="dash-panel-title">🤖 Sentiment Comparison</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            sent_cols = st.columns(n_tickers)
            for i, tkr in enumerate(compare_tickers):
                with sent_cols[i]:
                    news = cmp_fetch_news(tkr, max_results=10)
                    heads = [a['title'] for a in news]
                    if heads:
                        agg = cmp_get_agg(heads)
                    else:
                        agg = {'overall_score': 0.0, 'label': 'Neutral',
                               'bullish_pct': 0, 'bearish_pct': 0, 'neutral_pct': 100, 'count': 0}

                    score = agg['overall_score']
                    label = agg['label']
                    s_color = '#10b981' if label == 'Bullish' else '#ef4444' if label == 'Bearish' else '#f59e0b'

                    st.markdown(f"""
                    <div style="background:#111827;border:1px solid #1e2a3a;border-radius:8px;padding:16px;text-align:center;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;font-weight:700;color:{COMPARE_COLORS[i]};margin-bottom:8px;">{tkr}</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:700;color:{s_color};">{score:+.2f}</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:600;color:{s_color};text-transform:uppercase;letter-spacing:2px;margin-top:4px;">{label}</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#6b7280;margin-top:8px;">
                            ▲ {agg['bullish_pct']}% &nbsp; ● {agg['neutral_pct']}% &nbsp; ▼ {agg['bearish_pct']}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Train Models ──────────────────────────────────────────
            st.markdown("---")
            st.markdown("""
            <div class="dash-panel">
                <div class="dash-panel-header">
                    <span class="dash-panel-title">🎓 Training Models</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            compare_epochs = 25
            compare_forecast_days = 14
            all_results = {}

            for i, tkr in enumerate(compare_tickers):
                with st.spinner(f"Training {tkr} ({i+1}/{n_tickers})..."):
                    data_dict = prepare_data_pipeline(
                        ticker=tkr,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        seq_length=seq_length,
                        train_ratio=0.8
                    )

                    cmp_model = build_model(input_shape=(seq_length, data_dict['n_features']))
                    early_stop = create_early_stopping(patience=8)
                    train_model(
                        model=cmp_model,
                        X_train=data_dict['X_train'], y_train=data_dict['y_train'],
                        X_val=data_dict['X_test'], y_val=data_dict['y_test'],
                        epochs=compare_epochs, batch_size=batch_size,
                        callbacks=[early_stop]
                    )

                    y_pred = make_predictions(cmp_model, data_dict['X_test'])
                    y_test_orig = inverse_transform_close(data_dict['y_test'], data_dict['scaler'], data_dict['n_features'])
                    y_pred_orig = inverse_transform_close(y_pred, data_dict['scaler'], data_dict['n_features'])
                    cmp_metrics = evaluate_model(y_test_orig, y_pred_orig)

                    future_prices = predict_future(
                        model=cmp_model,
                        normalized_data=data_dict['normalized_data'],
                        scaler=data_dict['scaler'],
                        feature_df=data_dict['feature_df'],
                        seq_length=seq_length,
                        n_days=compare_forecast_days
                    )

                    all_results[tkr] = {
                        'metrics': cmp_metrics,
                        'future_prices': future_prices,
                        'original_df': data_dict['original_df'],
                    }

                st.success(f"✅ {tkr} — trained ({len(data_dict['original_df'])} days)")

            # ── Forecast Overlay Chart ────────────────────────────────
            st.markdown(f"""
            <div class="dash-panel">
                <div class="dash-panel-header">
                    <span class="dash-panel-title">🔮 Forecast Comparison — {compare_forecast_days} Trading Days</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig_compare = go.Figure()
            for i, tkr in enumerate(compare_tickers):
                res = all_results[tkr]
                orig_df = res['original_df']

                today_ts = pd.Timestamp(date.today())
                last_d = orig_df.index[-1]
                last_d_naive = last_d.tz_localize(None) if last_d.tzinfo is not None else last_d
                anchor_d = max(last_d_naive.normalize(), today_ts)
                f_dates = pd.bdate_range(start=anchor_d + pd.Timedelta(days=1), periods=compare_forecast_days)

                # Normalize to % change for fair comparison
                last_p = float(orig_df['Close'].iloc[-1])
                pct_change = ((res['future_prices'].flatten() - last_p) / last_p) * 100

                fig_compare.add_trace(go.Scatter(
                    x=f_dates, y=pct_change,
                    mode='lines+markers', name=tkr,
                    line=dict(color=COMPARE_COLORS[i], width=2),
                    marker=dict(size=4)
                ))

            fig_compare.add_hline(y=0, line_dash="dash", line_color="#6b7280", line_width=1)
            fig_compare.update_layout(
                title='Forecast: % Change from Last Close',
                xaxis_title='Date', yaxis_title='% Change',
                xaxis=dict(constrain='domain'),
                yaxis=dict(constrain='domain'),
                hovermode='x unified',
                template='plotly_dark',
                paper_bgcolor='#0a0e17',
                plot_bgcolor='#0a0e17',
                height=450,
                legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
                font=dict(family='JetBrains Mono, monospace')
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            # ── Metrics Comparison Table ──────────────────────────────
            st.markdown("""
            <div class="dash-panel">
                <div class="dash-panel-header">
                    <span class="dash-panel-title">🎯 Performance Comparison</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            metrics_data = {}
            for tkr in compare_tickers:
                m = all_results[tkr]['metrics']
                metrics_data[tkr] = {
                    'RMSE ($)': f"${m['RMSE']:.2f}",
                    'MAE ($)': f"${m['MAE']:.2f}",
                    'MAPE (%)': f"{m['MAPE']:.2f}%",
                    'R² Score': f"{m['R2']:.4f}",
                    'Dir. Accuracy': f"{m['Dir_Accuracy']:.1f}%",
                }

            comparison_df = pd.DataFrame(metrics_data)
            st.dataframe(comparison_df, use_container_width=True)

            # ── Best Performers ───────────────────────────────────────
            best_r2 = max(compare_tickers, key=lambda t: all_results[t]['metrics']['R2'])
            best_mape = min(compare_tickers, key=lambda t: all_results[t]['metrics']['MAPE'])
            best_dir = max(compare_tickers, key=lambda t: all_results[t]['metrics']['Dir_Accuracy'])

            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1e2a3a;border-radius:8px;padding:16px 20px;margin-top:12px;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;font-weight:600;color:#9ca3af;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">
                    🏆 Best Performers
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;line-height:2;color:#e5e7eb;">
                    <span style="color:#10b981;">Best Fit (R²):</span> <b>{best_r2}</b> — {all_results[best_r2]['metrics']['R2']:.4f}<br>
                    <span style="color:#3b82f6;">Lowest Error (MAPE):</span> <b>{best_mape}</b> — {all_results[best_mape]['metrics']['MAPE']:.2f}%<br>
                    <span style="color:#f59e0b;">Best Direction:</span> <b>{best_dir}</b> — {all_results[best_dir]['metrics']['Dir_Accuracy']:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error during comparison: {str(e)}")
            st.exception(e)

# ═══════════════════════════════════════════════════════════════════════
#  METHODOLOGY & LIMITATIONS (always visible)
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("📋 Methodology & Limitations", expanded=False):
    st.markdown("""
    <div style="font-family:var(--font-mono); font-size:0.85rem; line-height:1.9; color:#d1d5db;">

    <h4 style="color:#f59e0b; margin-bottom:8px;">🧠 Model Approach</h4>
    This application uses a <b>Long Short-Term Memory (LSTM)</b> neural network — a type of
    recurrent neural network designed to learn patterns in sequential data. The model is trained
    on historical stock prices along with 7 engineered features (Close, Volume, RSI, MACD,
    Signal Line, SMA-20, EMA-12) to predict future closing prices.

    <h4 style="color:#10b981; margin-top:16px; margin-bottom:8px;">✅ Strengths</h4>
    <ul style="margin:0; padding-left:20px;">
        <li><b>Real data pipeline</b> — pulls live market data via Yahoo Finance API</li>
        <li><b>Feature engineering</b> — uses technical indicators (RSI, MACD, SMA, EMA), not just raw price</li>
        <li><b>Sentiment analysis</b> — VADER NLP scoring of real-time news headlines adds a qualitative signal</li>
        <li><b>Regularization</b> — Dropout layers + early stopping prevent overfitting</li>
        <li><b>Evaluation metrics</b> — provides RMSE, MAE, MAPE, R², and directional accuracy for transparency</li>
    </ul>

    <h4 style="color:#ef4444; margin-top:16px; margin-bottom:8px;">⚠️ Known Limitations</h4>
    <ul style="margin:0; padding-left:20px;">
        <li><b>Trend-following bias</b> — LSTM models on price data tend to predict values close to the
            previous day's price, which inflates R² without necessarily predicting <i>direction changes</i></li>
        <li><b>Directional accuracy</b> — a Dir. Accuracy near or below 50% means the model is not
            reliably predicting whether the stock will go up or down</li>
        <li><b>Forecast divergence</b> — multi-day forecasts compound prediction errors; accuracy
            degrades significantly beyond 3–5 days</li>
        <li><b>No external shocks</b> — the model cannot anticipate earnings surprises, geopolitical
            events, Fed rate decisions, or other market-moving catalysts</li>
        <li><b>Survivorship bias</b> — training on a single stock's history doesn't capture
            market-wide regime changes or correlations</li>
        <li><b>Sentiment ≠ causation</b> — news sentiment often <i>reflects</i> price moves rather
            than predicting them (lagging indicator)</li>
    </ul>

    <h4 style="color:#6b7280; margin-top:16px; margin-bottom:8px;">📌 Disclaimer</h4>
    <span style="color:#9ca3af;">This project is built for <b>educational and portfolio demonstration purposes only</b>.
    It is NOT financial advice. Stock markets are inherently unpredictable, and no model —
    including those used by professional quantitative funds — can consistently predict prices
    with certainty. Always consult a qualified financial advisor before making investment decisions.</span>

    </div>
    """, unsafe_allow_html=True)

