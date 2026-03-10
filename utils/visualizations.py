"""
Visualization Module — Dark Dashboard Theme
Creates interactive charts using Plotly with plotly_dark template.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

# ── Shared layout settings for dark theme ────────────────────────────
_DARK_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='#0a0e17',
    plot_bgcolor='#0a0e17',
    font=dict(family='JetBrains Mono, monospace', color='#e5e7eb'),
    xaxis=dict(gridcolor='#1e2a3a', zerolinecolor='#1e2a3a', constrain='domain'),
    yaxis=dict(gridcolor='#1e2a3a', zerolinecolor='#1e2a3a', constrain='domain'),
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        font=dict(size=11),
        yanchor='top', y=0.99,
        xanchor='left', x=0.01
    ),
    hovermode='x unified',
)


def plot_stock_history(df, ticker):
    """Plot historical stock prices in dark theme."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#10b981', width=2),
        fill='tozeroy',
        fillcolor='rgba(16,185,129,0.06)',
    ))

    fig.update_layout(
        title=f'{ticker} Historical Stock Prices',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=450,
        **_DARK_LAYOUT
    )

    return fig


def plot_predictions(dates, actual, predicted, train_size):
    """Plot actual vs predicted prices in dark theme."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        mode='lines', name='Actual Price',
        line=dict(color='#10b981', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=predicted,
        mode='lines', name='Predicted Price',
        line=dict(color='#f59e0b', width=2, dash='dash')
    ))

    if train_size > 0 and train_size < len(dates):
        split_val = dates[train_size]
        if isinstance(split_val, pd.Timestamp):
            split_val = split_val.to_pydatetime()
        fig.add_vline(
            x=split_val,
            line_dash="dash", line_color="#6b7280",
            annotation_text="Prediction Start",
            annotation_position="top"
        )

    fig.update_layout(
        title='Actual vs Predicted Prices',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=450,
        **_DARK_LAYOUT
    )

    return fig


def plot_training_history(history):
    """Plot training and validation loss in dark theme."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1, len(history.history['loss']) + 1)),
        y=history.history['loss'],
        mode='lines', name='Training Loss',
        line=dict(color='#3b82f6', width=2)
    ))

    if 'val_loss' in history.history:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(history.history['val_loss']) + 1)),
            y=history.history['val_loss'],
            mode='lines', name='Validation Loss',
            line=dict(color='#f59e0b', width=2)
        ))

    fig.update_layout(
        title='Training Progress',
        xaxis_title='Epoch',
        yaxis_title='Loss (MSE)',
        height=350,
        **_DARK_LAYOUT
    )

    return fig


def _hint(text, color="#9ca3af"):
    """Render a small colored hint below a metric."""
    st.markdown(
        f'<div style="font-size:0.75rem;margin-top:-10px;padding:4px 0;color:{color};'
        f'font-family:\'JetBrains Mono\',monospace;">{text}</div>',
        unsafe_allow_html=True
    )


def display_metrics(metrics):
    """Display evaluation metrics in styled columns with quality hints."""
    col1, col2, col3, col4, col5 = st.columns(5)

    # ── RMSE ──
    rmse = metrics['RMSE']
    with col1:
        st.metric(
            label="RMSE",
            value=f"${rmse:.2f}",
            help="Root Mean Squared Error - Lower is better"
        )
        if rmse < 2:
            _hint("✅ Excellent", "#10b981")
        elif rmse < 5:
            _hint("👍 Good", "#10b981")
        elif rmse < 10:
            _hint("⚠️ Fair — moderate error", "#f59e0b")
        else:
            _hint("❌ High — unreliable", "#ef4444")

    # ── MAE ──
    mae = metrics['MAE']
    with col2:
        st.metric(
            label="MAE",
            value=f"${mae:.2f}",
            help="Mean Absolute Error - Average prediction error in dollars"
        )
        if mae < 1.5:
            _hint("✅ Excellent", "#10b981")
        elif mae < 4:
            _hint("👍 Good", "#10b981")
        elif mae < 8:
            _hint("⚠️ Fair — some deviation", "#f59e0b")
        else:
            _hint("❌ High deviation", "#ef4444")

    # ── MAPE ──
    mape = metrics['MAPE']
    with col3:
        st.metric(
            label="MAPE",
            value=f"{mape:.2f}%",
            help="Mean Absolute Percentage Error - Lower is better"
        )
        if mape < 2:
            _hint("✅ Excellent — < 2%", "#10b981")
        elif mape < 5:
            _hint("👍 Good — within 5%", "#10b981")
        elif mape < 10:
            _hint("⚠️ Fair — 5-10% off", "#f59e0b")
        else:
            _hint("❌ Poor — > 10%", "#ef4444")

    # ── R² Score ──
    r2 = metrics['R2']
    with col4:
        st.metric(
            label="R² Score",
            value=f"{r2:.4f}",
            help="Coefficient of Determination - 1.0 = perfect fit"
        )
        if r2 >= 0.98:
            _hint("✅ Excellent fit", "#10b981")
        elif r2 >= 0.95:
            _hint("👍 Strong fit", "#10b981")
        elif r2 >= 0.90:
            _hint("⚠️ Reasonable fit", "#f59e0b")
        else:
            _hint("❌ Weak — needs tuning", "#ef4444")

    # ── Directional Accuracy ──
    dir_acc = metrics.get('Dir_Accuracy', 0)
    with col5:
        st.metric(
            label="Dir. Accuracy",
            value=f"{dir_acc:.1f}%",
            help="% of days where predicted UP/DOWN matches actual"
        )
        if dir_acc >= 60:
            _hint("✅ Beats random (50%)", "#10b981")
        elif dir_acc >= 50:
            _hint("⚠️ Coin-flip level", "#f59e0b")
        else:
            _hint("❌ Below 50%", "#ef4444")


