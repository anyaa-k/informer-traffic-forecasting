import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib
from datetime import timedelta

from model import InformerPlus

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Internet Traffic Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Internet Traffic Forecasting using Informer")
st.write(
    """
    Aplikasi ini digunakan untuk memprediksi trafik internet
    berdasarkan data historis menggunakan model Informer.
    Upload file CSV yang memiliki kolom:

    - timestamp
    - traffic
    """
)

# =====================================================
# LOAD MODEL & SCALER
# =====================================================

@st.cache_resource
def load_model():

    model = InformerPlus(
        input_dim=4,
        pred_len=12
    )

    model.load_state_dict(
        torch.load(
            "informer_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model

@st.cache_resource
def load_scalers():

    traffic_scaler = joblib.load(
        "traffic_scaler.pkl"
    )

    roll_scaler = joblib.load(
        "roll_scaler.pkl"
    )

    return traffic_scaler, roll_scaler


# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        # ==========================================
        # BATASI DATA UNTUK EFISIENSI
        # ==========================================

        if len(df) > 5000:

            st.warning(
                f"""
                Dataset berisi {len(df):,} baris.

                Untuk mempercepat proses prediksi,
                sistem hanya menggunakan 5000 data
                terakhir.
                """
            )

            df = df.tail(5000).reset_index(drop=True)
        # ==========================================
        # VALIDASI FORMAT
        # ==========================================

        required_columns = [
            "timestamp",
            "traffic"
        ]

        if not all(col in df.columns for col in required_columns):

            st.error(
                """
                ❌ Format CSV tidak sesuai.

                File harus memiliki kolom:

                - timestamp
                - traffic
                """
            )

            st.stop()

        # ==========================================
        # PREPROCESSING
        # ==========================================

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df["traffic"] = pd.to_numeric(
            df["traffic"],
            errors="coerce"
        )

        df["traffic"] = (
            df["traffic"]
            .interpolate()
            .ffill()
            .bfill()
        )

        df["hour_sin"] = np.sin(
            2 * np.pi * df["timestamp"].dt.hour / 24
        )

        df["hour_cos"] = np.cos(
            2 * np.pi * df["timestamp"].dt.hour / 24
        )

        df["traffic_roll"] = (
            df["traffic"]
            .rolling(window=12)
            .mean()
            .ffill()
            .bfill()
        )

        if len(df) < 288:

            st.error(
                """
                ❌ Data tidak cukup.

                Minimal diperlukan 288 baris data
                untuk melakukan prediksi.
                """
            )

            st.stop()
        
        # ==========================================
        # DATASET INFORMATION
        # ==========================================

        st.subheader("📊 Dataset Information")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Rows",
            f"{len(df):,}"
        )

        col2.metric(
            "Start Date",
            str(df["timestamp"].min())[:16]
        )

        col3.metric(
            "End Date",
            str(df["timestamp"].max())[:16]
        )

        st.write(
            f"""
            Dataset contains **{len(df):,} traffic records**
            that will be used for forecasting.

            The model uses the latest **288 observations**
            as input to predict the next **12 traffic values**
            (approximately 60 minutes ahead).
            """
        )

        # ==========================================
        # LOAD SCALER
        # ==========================================

        traffic_scaler, roll_scaler = load_scalers()

        traffic_scaled = traffic_scaler.transform(
            df[["traffic"]]
        )

        roll_scaled = roll_scaler.transform(
            df[["traffic_roll"]]
        )

        other_features = df[
            ["hour_sin", "hour_cos"]
        ].values

        data_combined = np.hstack([
            traffic_scaled,
            roll_scaled,
            other_features
        ])

        # ==========================================
        # AMBIL 288 DATA TERAKHIR
        # ==========================================

        X_input = data_combined[-288:]

        X_input = torch.tensor(
            X_input,
            dtype=torch.float32
        ).unsqueeze(0)

        # ==========================================
        # PREDIKSI
        # ==========================================

        model = load_model()

        with torch.no_grad():

            pred_scaled = model(
                X_input
            ).numpy()[0]

        pred = traffic_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).flatten()

        forecast_mean = np.mean(pred)

        # ==========================================
        # TIMESTAMP HASIL PREDIKSI
        # ==========================================

        last_timestamp = df["timestamp"].iloc[-1]

        future_times = [
            last_timestamp + timedelta(minutes=5*(i+1))
            for i in range(12)
        ]

        result_df = pd.DataFrame({
            "Timestamp": future_times,
            "Predicted Traffic": pred
        })

        # ==========================================
        # RINGKASAN
        # ==========================================

        last_traffic = float(
            df["traffic"].iloc[-1]
        )
        st.write("Last Actual Traffic:", last_traffic)
        st.write("First Forecast:", pred[0])

        avg_prediction = float(
            np.mean(pred)
        )

        change_percent = (
            (avg_prediction - last_traffic)
            / last_traffic
        ) * 100

        trend = (
            "Increasing"
            if avg_prediction > last_traffic
            else "Decreasing"
        )

        st.success("✅ Prediction completed successfully")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Latest Traffic",
            f"{last_traffic:,.2f}"
        )

        col2.metric(
            "Average Forecast",
            f"{avg_prediction:,.2f}"
        )

        col3.metric(
            "Change",
            f"{change_percent:.2f}%"
        )

        # ==========================================
        # INTERPRETASI
        # ==========================================

        pred_min = np.min(pred)
        pred_max = np.max(pred)

        st.subheader("Prediction Analysis")

        if change_percent > 10:

            st.success(
                f"""
                Traffic is predicted to increase significantly.

                • Current traffic : {last_traffic:,.2f}

                • Average forecast : {avg_prediction:,.2f}

                • Expected increase : {change_percent:.2f}%

                This pattern indicates growing network demand.
                Additional bandwidth allocation may be required
                to prevent congestion.
                """
            )

        elif change_percent > 0:

            st.info(
                f"""
                Traffic is predicted to increase slightly.

                • Current traffic : {last_traffic:,.2f}

                • Average forecast : {avg_prediction:,.2f}

                • Expected increase : {change_percent:.2f}%

                Network utilization is expected to remain stable
                with moderate growth.
                """
            )

        elif change_percent > -10:

            st.info(
                f"""
                Traffic is predicted to decrease slightly.

                • Current traffic : {last_traffic:,.2f}

                • Average forecast : {avg_prediction:,.2f}

                • Expected decrease : {abs(change_percent):.2f}%

                Current network capacity should remain sufficient.
                """
            )

        else:

            st.warning(
                f"""
                Traffic is predicted to decrease significantly.

                • Current traffic : {last_traffic:,.2f}

                • Average forecast : {avg_prediction:,.2f}

                • Expected decrease : {abs(change_percent):.2f}%

                Resource utilization may decline during the
                forecast horizon.
                """
            )

        st.write(
            f"""
            Forecast Summary:

            • Minimum predicted traffic : {pred_min:,.2f}

            • Maximum predicted traffic : {pred_max:,.2f}

            • Average predicted traffic : {avg_prediction:,.2f}

            • Forecast horizon : 12 steps (60 minutes)

            • Predicted trend : {trend}
            """
        )

        # ==========================================
        # HISTORICAL DATA
        # ==========================================

        st.subheader("📈 Historical Traffic")

        history_display = df[
            ["timestamp", "traffic"]
        ].tail(288)

        fig_hist, ax_hist = plt.subplots(figsize=(12,5))

        ax_hist.plot(
            history_display["timestamp"],
            history_display["traffic"],
            linewidth=2,
            label="Historical Traffic"
        )

        ax_hist.set_title(
            "Historical Internet Traffic (Last 24 Hours)"
        )

        ax_hist.set_xlabel("Time")
        ax_hist.set_ylabel("Traffic Volume")

        ax_hist.grid(True, alpha=0.3)

        ax_hist.legend()

        st.pyplot(fig_hist)

        # ==========================================
        # HISTORICAL + FORECAST
        # ==========================================

        st.subheader(
            "📊 Historical Traffic and Forecast"
        )

        history_part = df[
            ["timestamp", "traffic"]
        ].tail(288)

        forecast_part = pd.DataFrame({
            "timestamp": future_times,
            "traffic": pred
        })

        fig, ax = plt.subplots(figsize=(12,5))

        ax.plot(
            history_part["timestamp"],
            history_part["traffic"],
            linewidth=2,
            color="#1f77b4",
            label="Historical Traffic"
        )

        forecast_x = [history_part["timestamp"].iloc[-1]] + future_times
        forecast_y = [history_part["traffic"].iloc[-1]] + list(pred)

        ax.plot(
            forecast_x,
            forecast_y,
            color="#ff7f0e",
            linewidth=2,
            label="Forecast"
        )

        # Penanda awal forecast
        ax.axvline(
            x=history_part["timestamp"].iloc[-1],
            linestyle=":",
            linewidth=2,
            label="Forecast Start"
        )

        ax.set_title(
            "Internet Traffic Forecast"
        )

        ax.set_xlabel("Time")
        ax.set_ylabel("Traffic Volume")

        ax.grid(True, alpha=0.3)

        ax.legend()

        st.pyplot(fig)

        # ==========================================
        # TABEL HASIL
        # ==========================================

        st.subheader(
            "Forecast Results"
        )

        st.dataframe(
            result_df,
            use_container_width=True
        )

        csv_download = result_df.to_csv(
            index=False
        )

        st.download_button(
            label="📥 Download Forecast Results",
            data=csv_download,
            file_name="forecast_result.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"Terjadi kesalahan: {str(e)}"
        )