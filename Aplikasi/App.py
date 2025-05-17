import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ✅ Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Harga Komoditas", layout="wide", page_icon="🛢️")

# 🔄 Load dan Preprocessing Data
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df = df[["Tanggal", "Harga"]]
        df["Tanggal"] = pd.to_datetime(df["Tanggal"],format="%Y/%m/%d", errors='coerce')
        df = df.dropna(subset=["Tanggal", "Harga"])
        df["Harga"] = pd.to_numeric(df["Harga"], errors='coerce')
        df = df.dropna()
        df.set_index("Tanggal", inplace=True)
        df.sort_index(inplace=True)
        df = df.asfreq('D')
        df["Harga"] = df["Harga"].interpolate(method='linear')

        if len(df) < 100:
            st.warning("Data terlalu sedikit. Minimal 100 data diperlukan.")
            return None
        return df
    except Exception as e:
        st.warning(f"Format data tidak sesuai. Pastikan kolom 'Tanggal' dan 'Harga' tersedia.")
        return None

#  Training Model ARIMA
@st.cache_resource
def train_and_evaluate_model(df):
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    model = ARIMA(train['Harga'], order=(0, 1, 0), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()

    history = list(train['Harga'])
    predictions = []
    actual_values = list(test['Harga'])

    for t in range(len(test)):
        model = ARIMA(history, order=(0, 1, 0), enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(actual_values[t])

    mae = float(mean_absolute_error(actual_values, predictions))
    rmse = float(np.sqrt(mean_squared_error(actual_values, predictions)))
    mean_actual = np.mean(actual_values)
    mape_percentage = (mae / mean_actual) * 100

    return model_fit, mae, rmse, mape_percentage

#  Prediksi Harga Masa Depan
def predict(start, end, df, model_fit):
    selisih_hari = (end - start).days + 1
    forecast = model_fit.forecast(steps=selisih_hari)
    forecast = np.array(forecast).ravel()

    last_date = df.index[-1]
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, selisih_hari + 1)]
    forecast_df = pd.DataFrame({"Tanggal": forecast_dates, "Prediksi Harga": forecast})
    forecast_df.set_index("Tanggal", inplace=True)

    st.subheader(f"📈 Prediksi Harga {selisih_hari} Hari Kedepan")
    with st.expander("🔍 Lihat Detail Prediksi"):
        st.dataframe(forecast_df, width=600)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Harga"], mode='lines+markers', name='Data Historis', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Prediksi Harga"], mode='lines+markers', name='Prediksi', line=dict(color='red', dash='dash')))
    fig.update_layout(title="Prediksi Harga Komoditi", xaxis_title="Tanggal", yaxis_title="Harga (Rp)", hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

# 📊 Evaluasi Model
def evaluate_model(mae, rmse, mape_percentage):
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.metric("MAE", f"{mae:.1f}")
    with col2:
        st.metric("RMSE", f"{rmse:.1f}")
    with col3:
        st.metric("MAPE", f"{mape_percentage:.1f}%")

# 🧾 Upload Data Excel
st.title("Forecasting Harga Komoditas Menggunakan ARIMA")

uploaded_file = st.file_uploader("📂 Upload File Excel", type=".xlsx", help="Pastikan memiliki kolom 'Tanggal' dan 'Harga'")

if uploaded_file is None:
    st.warning("Silakan upload file terlebih dahulu untuk mulai.")
    st.stop()

df = load_data(uploaded_file)
if df is None or df.empty:
    st.stop()

with st.sidebar:
    st.subheader("📌 Data Terupload")
    st.dataframe(df, use_container_width=True)

today = pd.Timestamp.today().date()
last_data_date = df.index[-1].date()
start_date = today if today > last_data_date else last_data_date

st.sidebar.subheader("📅 Pilih Tanggal Prediksi")
start = st.sidebar.date_input("Tanggal Mulai", value=start_date, disabled=True)
end = st.sidebar.date_input("Tanggal Selesai", value=None, min_value=start_date + pd.Timedelta(days=1))

if st.sidebar.button("Prediksi!", use_container_width=True):
    if end is None:
        st.warning("Silakan pilih tanggal selesai.")
    elif end <= start:
        st.warning("Tanggal selesai harus lebih besar dari tanggal mulai!")
    else:
        model_fit, mae, rmse, mape_percentage = train_and_evaluate_model(df)
        evaluate_model(mae, rmse, mape_percentage)
        predict(start, end, df, model_fit)
