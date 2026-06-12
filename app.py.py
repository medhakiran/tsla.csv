import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt

# Set up Streamlit page configuration
st.set_page_config(page_title="Tesla (TSLA) Stock Price Predictor", layout="wide")

# =====================================================================
# 1. NATIVE TIME-SERIES RECURRENT NEURAL NETWORK DEFINITION
# =====================================================================
class NativeTimeRNN:
    def __init__(self, input_dim=1, hidden_dim1=32, hidden_dim2=16, output_dim=1, lr=0.01, dropout_rate=0.2):
        self.hd1, self.hd2, self.lr, self.dropout_rate = hidden_dim1, hidden_dim2, lr, dropout_rate
        
        # Xavier-like weight initialization for time-series sequences
        self.W_xh1 = np.random.randn(hidden_dim1, input_dim) * np.sqrt(2.0 / input_dim)
        self.W_hh1 = np.random.randn(hidden_dim1, hidden_dim1) * np.sqrt(2.0 / hidden_dim1)
        self.b_h1 = np.zeros((hidden_dim1, 1))
        
        self.W_hh2 = np.random.randn(hidden_dim2, hidden_dim1) * np.sqrt(2.0 / hidden_dim1)
        self.W_h2h2 = np.random.randn(hidden_dim2, hidden_dim2) * np.sqrt(2.0 / hidden_dim2)
        self.b_h2 = np.zeros((hidden_dim2, 1))
        
        self.W_hy = np.random.randn(output_dim, hidden_dim2) * np.sqrt(2.0 / hidden_dim2)
        self.b_y = np.zeros((output_dim, 1))

    def forward(self, X_seq, is_training=False):
        T = X_seq.shape[0]
        h1_states = {-1: np.zeros((self.hd1, 1))}
        h2_states = {-1: np.zeros((self.hd2, 1))}
        
        for t in range(T):
            x_t = X_seq[t].reshape(-1, 1)
            h1_t = np.tanh(np.dot(self.W_xh1, x_t) + np.dot(self.W_hh1, h1_states[t-1]) + self.b_h1)
            if is_training:
                mask = (np.random.rand(*h1_t.shape) >= self.dropout_rate) / (1.0 - self.dropout_rate)
                h1_t *= mask
            h1_states[t] = h1_t
            
            h2_t = np.tanh(np.dot(self.W_hh2, h1_states[t]) + np.dot(self.W_h2h2, h2_states[t-1]) + self.b_h2)
            h2_states[t] = h2_t
            
        y_out = np.dot(self.W_hy, h2_states[T-1]) + self.b_y
        return h2_states[T-1]

    def fit_sequence_batch(self, X_batches, y_batches, epochs=5):
        for _ in range(epochs):
            for X_s, y_s in zip(X_batches, y_batches):
                last_hidden = self.forward(X_s, is_training=True)
                pred = np.dot(self.W_hy, last_hidden) + self.b_y
                error = pred - y_s.reshape(-1, 1)
                
                self.W_hy -= self.lr * np.dot(error, last_hidden.T)
                self.b_y -= self.lr * error

# =====================================================================
# 2. TIME-SERIES PREPROCESSING ENGINE (SLIDING WINDOW ENGINE)
# =====================================================================
def prepare_sliding_windows(data, window_size, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size + forecast_horizon - 1])
    return np.array(X), np.array(y)

# =====================================================================
# 3. STREAMLIT INTERFACE AND LIVE YFINANCE CONNECTIVITY
# =====================================================================
st.title("⚡ Tesla (TSLA) Stock Price Prediction Dashboard")
st.markdown("### Deep Learning Time-Series Prediction Engine (SimpleRNN / Stacked Architecture)")

# Sidebar configurations
st.sidebar.header("🔧 Model Configuration")
horizon_selection = st.sidebar.selectbox("Select Forecast Horizon", [1, 5, 10], format_func=lambda x: f"{x} Day(s) Ahead Prediction")
window_days = st.sidebar.slider("Lookback Window Size (Days)", min_value=10, max_value=60, value=30)
training_epochs = st.sidebar.slider("Training Epochs", min_value=5, max_value=30, value=15)

st.markdown("---")

# Fetch live TSLA stock data using yfinance
with st.spinner("Fetching live stock data from Yahoo Finance API..."):
    tsla_df = yf.download("TSLA", start="2023-01-01")

if not tsla_df.empty:
    st.success("Successfully connected to Yahoo Finance database.")
    
    # CRITICAL FIX 1: Explicitly flatten yfinance's modern MultiIndex column structure
    if isinstance(tsla_df.columns, pd.MultiIndex):
        tsla_df.columns = tsla_df.columns.get_level_values(0)
        
    target_col = 'Adj Close' if 'Adj Close' in tsla_df.columns else 'Close'
    
    # CRITICAL FIX 2: Replaced removed fillna(method='ffill') syntax with native .ffill()
    data_series = tsla_df[[target_col]].ffill().values
    
    # Feature Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_series)
    
    # Generate input-output sequences
    X_seq, y_seq = prepare_sliding_windows(scaled_data, window_days, horizon_selection)
    
    # Train-test split
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    # Initialize and train model
    model = NativeTimeRNN(input_dim=1, hidden_dim1=32, hidden_dim2=16, output_dim=1, lr=0.01)
    
    with st.spinner("Optimizing Deep Network Weights via Sequence Backpropagation..."):
        model.fit_sequence_batch(X_train, y_train, epochs=training_epochs)
        
    # Evaluate performance on test data
    test_predictions = []
    for test_sample in X_test:
        hidden_out = model.forward(test_sample, is_training=False)
        pred_val = np.dot(model.W_hy, hidden_out) + model.b_y
        test_predictions.append(pred_val.flatten())
        
    test_predictions = np.array(test_predictions)
    
    # Invert scaling back to original dollar values
    actual_prices = scaler.inverse_transform(y_test)
    predicted_prices = scaler.inverse_transform(test_predictions)
    
    # Calculate Mean Squared Error (MSE)
    mse_metric = np.mean((actual_prices - predicted_prices) ** 2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Performance (Test MSE)", f"{mse_metric:.4f}")
    with col2:
        st.metric("Target Horizon Configuration", f"{horizon_selection} Day(s) Ahead")
        
    # =====================================================================
    # 4. DATA VISUALIZATION CHART
    # =====================================================================
    st.markdown("### 📊 Actual vs. Predicted Trend Line Chart")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    test_dates = tsla_df.index[-len(actual_prices):]
    
    ax.plot(test_dates, actual_prices, label="Actual TSLA Price", color="#1f77b4", linewidth=2)
    ax.plot(test_dates, predicted_prices, label=f"Predicted Price ({horizon_selection}-Day Out)", color="#ff7f0e", linestyle="--", linewidth=2)
    
    ax.set_title(f"Tesla Inc. Price Forecast (Horizon: {horizon_selection} Day(s))", fontsize=14, fontweight='bold')
    ax.set_xlabel("Timeline", fontsize=12)
    ax.set_ylabel("Price in USD ($)", fontsize=12)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(rotation=45)
    
    st.pyplot(fig)
    
    # =====================================================================
    # 5. BUSINESS ACTION METRICS
    # =====================================================================
    st.markdown("### 💡 Algorithmic Trading Signal Insights")
    last_actual = actual_prices[-1][0]
    last_predicted = predicted_prices[-1][0]
    percentage_change = ((last_predicted - last_actual) / last_actual) * 100
    
    if percentage_change > 1.5:
        st.info(f"📈 **Bullish Signal Detected:** Model projects an upward movement of **{percentage_change:.2f}%** over the target horizon. Strategy recommends positioning for a potential **BUY** execution.")
    elif percentage_change < -1.5:
        st.warning(f"📉 **Bearish Signal Detected:** Model projects a downward contraction of **{percentage_change:.2f}%** over the target horizon. Strategy recommends positioning for a potential **SHORT/SELL** risk containment protocol.")
    else:
        st.success(f"↔️ **Sideways Market Signal:** Predicted movement is minimal (**{percentage_change:.2f}%** Change). Strategy advises a **HOLD** posture with tight volatility risk containment parameters.")
else:
    st.error("Unable to collect market telemetry from yfinance data endpoints. Check networking pathways.")

    # Run this cell inside your notebook to start your engine back up:
import subprocess
import sys

process = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
print("Server restarted successfully! Check http://localhost:8501")