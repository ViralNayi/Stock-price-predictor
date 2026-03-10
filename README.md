# 📈 Stock Price Prediction — Intelligence Dashboard

A full-featured **Streamlit** web application that forecasts stock prices using **LSTM neural networks**, enriched with **real-time sentiment analysis** and **technical indicators**. Built with a sleek, dark "intelligence monitor" UI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Category | Details |
|----------|---------|
| **LSTM Forecasting** | 3-layer LSTM (50 units each) with Dropout regularisation and early stopping |
| **7-Feature Input** | Close, Volume, RSI-14, MACD, MACD Signal, SMA-20, EMA-12 |
| **Sentiment Analysis** | VADER NLP scoring of live news headlines (Bullish / Bearish / Neutral) |
| **News Feed** | Dual-source: NewsAPI.org + Google News RSS fallback |
| **Technical Indicators** | Real-time RSI, MACD, SMA-20, EMA-12 with interpretive hints |
| **Compare Stocks** | Train & compare 2–3 tickers side-by-side (sentiment, metrics, forecast overlay) |
| **Future Forecast** | Iterative multi-day forecast with on-the-fly indicator recomputation |
| **Live Price Header** | Real-time price, daily change, and live status indicator |
| **Performance Metrics** | RMSE, MAE, MAPE, R², Directional Accuracy — with quality-level hints |
| **Export** | Download forecast and prediction results as CSV |
| **Model Persistence** | Save/load trained Keras models to disk |
| **Dark Dashboard UI** | Custom CSS with JetBrains Mono typography, glassmorphism cards, and micro-animations |

---

## 🧠 Why LSTM?

Stock prices are **sequential data with temporal dependencies** — today's price is influenced by patterns from weeks ago.

| Model | Fit? | Reasoning |
|-------|:----:|----|
| **LSTM** ✅ | ✅ | Built for sequences; memory gates selectively retain/forget past patterns |
| **GRU** | ✅ | Simpler LSTM variant — faster but similar accuracy |
| **Simple RNN** | ❌ | Vanishing gradient problem — forgets beyond ~10 steps |
| **Linear Regression** | ❌ | Can't capture non-linear or temporal dependencies |
| **Random Forest / XGBoost** | ⚠️ | Strong for tabular data, but ignores sequence order |
| **Transformer** | ✅ | State-of-the-art, but harder to train and explain |
| **ARIMA** | ⚠️ | Classic time-series, limited to single-feature input |

---

## 🏗️ Project Structure

```
stock-price-predictor/
│
├── app.py                        # Main Streamlit dashboard (973 lines)
│
├── model/
│   ├── data_processor.py         # Data fetching, preprocessing, technical indicators, normalization
│   ├── lstm_model.py             # LSTM model architecture (build, save, load, early stopping)
│   └── trainer.py                # Training loop, evaluation metrics, future prediction
│
├── sentiment/
│   ├── __init__.py               # Package init
│   ├── analyzer.py               # VADER-based sentiment scoring (single, batch, aggregate)
│   └── news_fetcher.py           # Dual-source news: NewsAPI + Google News RSS fallback
│
├── utils/
│   └── visualizations.py         # Plotly charts (history, predictions, training loss, metrics)
│
├── styles/
│   └── dashboard.css             # Custom dark-theme CSS (variables, cards, animations)
│
├── saved_models/                 # Trained .h5/.keras models (git-ignored)
├── .streamlit/
│   ├── config.toml               # Streamlit theme config
│   └── secrets.toml              # NewsAPI key (git-ignored)
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Standard ignores for Python/ML projects
└── README.md                     # This file
```

---

## 🎓 Model Architecture

```
Input  →  [seq_length × 7 features]
           │
           ▼
      LSTM (50 units, return_sequences=True)
      Dropout (0.2)
           │
      LSTM (50 units, return_sequences=True)
      Dropout (0.2)
           │
      LSTM (50 units)
      Dropout (0.2)
           │
      Dense (1)  →  Next-day Close price
```

- **Optimizer**: Adam
- **Loss**: Mean Squared Error
- **Regularisation**: Dropout (20%) + EarlyStopping (patience=10)

---

## 📋 Requirements

- Python 3.10+
- TensorFlow 2.16+
- See [`requirements.txt`](requirements.txt) for the full list

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/stock-price-predictor.git
   cd stock-price-predictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Add a NewsAPI key** for richer news data
   ```bash
   mkdir .streamlit
   echo '[newsapi]' > .streamlit/secrets.toml
   echo 'api_key = "YOUR_NEWSAPI_KEY"' >> .streamlit/secrets.toml
   ```
   > Without a key the app falls back to Google News RSS — works out of the box.

---

## 🎯 Usage

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Workflow

1. **Select a ticker** from the dropdown (or type a custom symbol)
2. Browse **live price**, **sentiment gauge**, **news feed**, and **technical indicators**
3. Adjust model parameters in the sidebar (sequence length, epochs, batch size)
4. Click **🚀 TRAIN & PREDICT** to train the LSTM and see results
5. View **predictions vs actual**, **future forecast**, and **performance metrics**
6. Expand **📊 Compare Stocks** to train and compare 2–3 tickers side-by-side
7. Download forecast or prediction CSVs with the export buttons

---

## 📊 Key Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| **RMSE** | Root Mean Squared Error (in $) | < $5 |
| **MAE** | Mean Absolute Error (in $) | < $4 |
| **MAPE** | Mean Absolute Percentage Error | < 5% |
| **R² Score** | Coefficient of determination (1.0 = perfect) | > 0.95 |
| **Dir. Accuracy** | % of days where up/down direction was correct | > 55% |

---

## ⚠️ Limitations & Disclaimer

- **Trend-following bias** — LSTM models tend to lag behind actual price, inflating R² without predicting direction changes
- **Forecast divergence** — multi-day forecasts compound errors; accuracy degrades beyond 3–5 days
- **No external shocks** — the model cannot anticipate earnings, geopolitical events, or Fed decisions
- **Sentiment ≠ causation** — news sentiment often reflects price moves rather than predicting them

> **This project is for educational and portfolio demonstration purposes only. It is NOT financial advice.** Always consult a qualified financial advisor before making investment decisions.

---

## 🛡️ License

MIT License — feel free to use for learning and portfolio projects.

---

## 👨‍💻 Built With

- **Deep Learning** — TensorFlow / Keras LSTM networks
- **NLP** — VADER sentiment analysis
- **Data Engineering** — yfinance, Pandas, NumPy, scikit-learn
- **Visualisation** — Plotly interactive charts
- **Frontend** — Streamlit + custom CSS dark theme
