import streamlit as st
from utils.data_loader import load_main_data, load_info_data, create_master_station_data, get_station_data
import pandas as pd
import numpy as np
import base64
import plotly.express as px
from utils.footer import show_footer
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import io
import os

# Cache
data = load_main_data()

# Get current data (last row)
current_data = data.iloc[-1]
prev_data = data.iloc[-2]

# Extract current values and calculate deltas
current_katulampa = current_data.get('Katulampa (air)')
current_depok = current_data.get('Depok (air)') 
current_manggarai = current_data.get('Manggarai (air)')

delta_katulampa = current_katulampa - prev_data.get('Katulampa (air)', current_katulampa)
delta_depok = current_depok - prev_data.get('Depok (air)', current_depok)
delta_manggarai = current_manggarai - prev_data.get('Manggarai (air)', current_manggarai)

# Get weather info
weather_katulampa = current_data.get('Katulampa (cuaca)', 'Data tidak tersedia')
weather_depok = current_data.get('Depok (cuaca)', 'Data tidak tersedia')
weather_manggarai = current_data.get('Manggarai (cuaca)', 'Data tidak tersedia')

# Get last updated time
last_updated = current_data.get('Tanggal', 'Tidak diketahui')

with open("assets/images/logo_full.svg", "rb") as img_file:
    logo_base64 = base64.b64encode(img_file.read()).decode()

st.markdown(f"""
<div style="text-align: center; padding: 0 !important">
    <img src="data:image/svg+xml;base64,{logo_base64}" width="400">
    <h1>Pintu Air Manggarai</h1>
</div>
""", unsafe_allow_html=True)
st.markdown("Website ini dikhususkan untuk **monitoring dan prediksi** tinggi muka air di **Pintu Air Manggarai**. Data dari Katulampa dan Depok ditampilkan sebagai **indikator upstream** yang mempengaruhi kondisi di Manggarai.")
st.markdown("**Pintu Air Manggarai** berada pada aliran **Sungai Ciliwung**. Terdapat beberapa pintu air di sepanjang aliran Sungai Ciliwung. Dimulai dari paling hulu sungai, yaitu Pintu Air Katulampa -> Pintu Air Depok -> **Pintu Air Manggarai**")

def get_alert_level(location, height):
    station = get_station_data(location)
    
    if height >= station['siaga_1']:
        return "🔴 Siaga 1", "error"
    elif height >= station['siaga_2']:
        return "🟠 Siaga 2", "warning" 
    elif height >= station['siaga_3']:
        return "🟡 Siaga 3", "info"
    else:
        return "🟢 Normal", "success"

def load_prediction_model():
    """Load the trained XGBoost model"""
    
    # Gak usah initialize dengan parameter
    model = xgb.XGBRegressor()
    
    model.load_model('models/13_best_model.json')
    
    # Get parameter dari model yang sudah di-load
    model_params = model.get_params()
    
    return model, model_params

def get_threshold_info(location):

    df = load_info_data()
    row = df[df['Lokasi'] == location].iloc[0]
    
    return {
        'name': row['Lokasi'],
        'siaga3': f"{row['Siaga 3']} cm",
        'siaga2': f"{row['Siaga 2']} cm", 
        'siaga1': f"{row['Siaga 1']} cm",
        'latitude': row['Latitude'],
        'longitude': row['Longitude']
    }

@st.cache_resource
def create_prediction_data():
    """Create prediction data"""
    model, model_params = load_prediction_model()
    df = load_main_data()
    
    if 'Tanggal' in df.columns:
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df.set_index('Tanggal', inplace=True)
    
    feature_columns = [
        "time_index","hour_sin","hour_cos","dayofyear_sin","dayofyear_cos","manggarai_air_lag1","manggarai_air_lag2","manggarai_air_lag3","manggarai_air_lag4","manggarai_air_lag5","manggarai_air_lag6","manggarai_air_lag7","manggarai_air_lag9","manggarai_air_lag12","manggarai_air_lag14","manggarai_air_lag15","manggarai_air_lag17","manggarai_air_lag20","manggarai_air_lag21","depok_air_lag3","depok_air_lag6","depok_air_lag7","depok_air_lag8","depok_air_lag9","depok_air_lag10","depok_air_lag12","depok_air_lag13","depok_air_lag14","depok_air_lag16","depok_air_lag19","depok_air_lag20","depok_air_lag23","depok_air_lag24","katulampa_air_lag3","katulampa_air_lag4","katulampa_air_lag5","katulampa_air_lag14","katulampa_air_lag19","katulampa_air_lag21","manggarai_cuaca_lag1_hujan","manggarai_cuaca_lag2_hujan","manggarai_cuaca_lag4_hujan","manggarai_cuaca_lag5_hujan","manggarai_cuaca_lag7_hujan","manggarai_cuaca_lag9_hujan","manggarai_cuaca_lag12_hujan","manggarai_cuaca_lag19_hujan","depok_cuaca_lag1_hujan","depok_cuaca_lag2_hujan","depok_cuaca_lag4_hujan","depok_cuaca_lag5_hujan","depok_cuaca_lag6_hujan","depok_cuaca_lag7_hujan","depok_cuaca_lag9_hujan","depok_cuaca_lag22_hujan","katulampa_cuaca_lag14_hujan","katulampa_cuaca_lag18_hujan"
    ]
        
    # Prediction logic
    def predict_6_hours(last_row, model):
        current_features = [
            np.sin(2 * np.pi * last_row.name.hour / 24),
            np.cos(2 * np.pi * last_row.name.hour / 24),
            np.sin(2 * np.pi * last_row.name.dayofyear / 365.25),
            np.cos(2 * np.pi * last_row.name.dayofyear / 365.25)
        ]
        
        # Add your lag features
        for col in feature_columns[4:]:
            current_features.append(last_row.get(col, 0))
        
        prediction = model.predict([current_features])[0]
        return current_features, prediction
    
    # Predict next 6 hours
    last_data_point = df.iloc[-1]
    six_hour_predictions = []
    current_features = None
    last_prediction = last_data_point['Manggarai (air)']

    for hour in range(1, 7):
        next_time = last_data_point.name + pd.Timedelta(hours=hour)
        
        # Persiapan fitur untuk prediksi
        current_features, prediction = predict_6_hours(last_data_point, model)
        
        # Update fitur lag
        current_features[0] = np.sin(2 * np.pi * next_time.hour / 24)
        current_features[1] = np.cos(2 * np.pi * next_time.hour / 24)
        
        # Geser lag features
        for i in range(4, 22):  # Manggarai air lag features
            current_features[i] = current_features[i-1]
        current_features[4] = last_prediction  # Update manggarai_air_lag1
        
        # Set semua cuaca lag menjadi 1 (hujan)
        for i in range(42, len(current_features)):
            current_features[i] = 1
        
        # Prediksi dengan fitur yang diperbarui
        prediction = model.predict([current_features])[0]
        
        six_hour_predictions.append({
            'Jam Ke-': hour,
            'Waktu': next_time.strftime('%H:%M'),
            'Tanggal': next_time.strftime('%Y-%m-%d'),
            'Prediksi (cm)': prediction,
            'Range Min (cm)': prediction - 10,
            'Range Max (cm)': prediction + 10
        })
        
        # Update untuk iterasi selanjutnya
        last_prediction = prediction
    
    forecast_df = pd.DataFrame(six_hour_predictions)
    
    return forecast_df

# Sidebar File Upload
st.sidebar.header("Upload CSV for Prediction")
st.sidebar.download_button(
    label="Download Template CSV",
    data=open('data/template.csv', 'rb').read(),
    file_name='prediction_template.csv',
    mime='text/csv'
)
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV
    df = pd.read_csv(uploaded_file)
    
    # Load model
    model, _ = load_prediction_model()
    
    # Predict
    feature_columns = [
         "time_index","hour_sin","hour_cos","dayofyear_sin","dayofyear_cos","manggarai_air_lag1","manggarai_air_lag2","manggarai_air_lag3","manggarai_air_lag4","manggarai_air_lag5","manggarai_air_lag6","manggarai_air_lag7","manggarai_air_lag9","manggarai_air_lag12","manggarai_air_lag14","manggarai_air_lag15","manggarai_air_lag17","manggarai_air_lag20","manggarai_air_lag21","depok_air_lag3","depok_air_lag6","depok_air_lag7","depok_air_lag8","depok_air_lag9","depok_air_lag10","depok_air_lag12","depok_air_lag13","depok_air_lag14","depok_air_lag16","depok_air_lag19","depok_air_lag20","depok_air_lag23","depok_air_lag24","katulampa_air_lag3","katulampa_air_lag4","katulampa_air_lag5","katulampa_air_lag14","katulampa_air_lag19","katulampa_air_lag21","manggarai_cuaca_lag1_hujan","manggarai_cuaca_lag2_hujan","manggarai_cuaca_lag4_hujan","manggarai_cuaca_lag5_hujan","manggarai_cuaca_lag7_hujan","manggarai_cuaca_lag9_hujan","manggarai_cuaca_lag12_hujan","manggarai_cuaca_lag19_hujan","depok_cuaca_lag1_hujan","depok_cuaca_lag2_hujan","depok_cuaca_lag4_hujan","depok_cuaca_lag5_hujan","depok_cuaca_lag6_hujan","depok_cuaca_lag7_hujan","depok_cuaca_lag9_hujan","depok_cuaca_lag22_hujan","katulampa_cuaca_lag14_hujan","katulampa_cuaca_lag18_hujan"
    ]
    
    # Perform prediction
    predictions = model.predict(df[feature_columns])
    df['predicted_manggarai_air'] = predictions
    
    # Create download button
    csv = df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download predicted CSV",
        data=csv,
        file_name='predicted_data.csv',
        mime='text/csv',
    )

# Main Prediction Section
st.subheader("🎯 Prediksi Jangka Pendek (6 Jam Ke Depan)")

# Create forecast DataFrame
forecast_df = create_prediction_data()

# Display Table
st.dataframe(forecast_df, use_container_width=True)

# Plotting
fig = go.Figure()

# Plot configuration
fig.add_trace(go.Scatter(
    x=[0, 1, 2, 3, 4, 5],
    y=forecast_df['Prediksi (cm)'],
    mode='lines+markers',
    name='Prediksi 6 Jam',
    line=dict(color='red')
))

# Confidence Interval
fig.add_trace(go.Scatter(
    x=[0, 1, 2, 3, 4, 5],
    y=forecast_df['Range Max (cm)'],
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=[0, 1, 2, 3, 4, 5],
    y=forecast_df['Range Min (cm)'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(width=0),
    name='Confidence Interval'
))

# Alert Level Lines
alert_levels = {
    'Siaga 1': 950,
    'Siaga 2': 850,
    'Siaga 3': 750
}

for level, height in alert_levels.items():
    fig.add_hline(
        y=height, 
        line_dash='dash', 
        line_color='red',
        annotation_text=f"{level} ({height} cm)"
    )

fig.update_layout(
    title='Prediksi 6 Jam Ke Depan',
    xaxis_title='Jam Ke-',
    yaxis_title='Tinggi Muka Air (cm)',
    height=400
)

st.plotly_chart(fig, use_container_width=True)
    
with st.expander("ℹ️ Informasi Ambang Batas Siaga Manggarai"):
    st.markdown("""
    **Tinggi Muka Air Manggarai - Ambang Batas Siaga:**
    
    🟢 **Normal**: < 750 cm  
    🟡 **Siaga 3**: ≥ 750 cm  
    🟠 **Siaga 2**: ≥ 850 cm  
    🔴 **Siaga 1**: ≥ 950 cm  
    
    *Garis horizontal pada plot menunjukkan batas-batas siaga ini*
    """)

st.markdown("---")


# =========================
# LOAD DATA LANGSUNG DARI CSV
# =========================
train_df = pd.read_csv("data/90_train_predict.csv", parse_dates=["Tanggal"])
test_df  = pd.read_csv("data/90_test_predict.csv", parse_dates=["Tanggal"])

train_df = train_df.sort_values("Tanggal")
test_df  = test_df.sort_values("Tanggal")

# =========================
# UI CONTROL
# =========================
st.markdown("#### 🎛️ Informasi dataset dan model")

visible_traces = st.multiselect(
    "Tampilkan komponen grafik:",
    [
        "Data Aktual (Training)",
        "Prediksi (Training)",
        "Data Aktual (Testing)",
        "Prediksi (Testing)",
    ],
    default=[
        "Data Aktual (Training)",
        "Prediksi (Training)",
        "Data Aktual (Testing)",
        "Prediksi (Testing)",
    ]
)

def is_visible(name):
    return name in visible_traces

# =========================
# CREATE FIGURE
# =========================
fig = go.Figure()

# TRAINING
fig.add_trace(go.Scatter(
    x=train_df["Tanggal"],
    y=train_df["y_actual"],
    mode="lines",
    name="Data Aktual (Training)",
    visible=is_visible("Data Aktual (Training)"),
    line=dict(color="royalblue", width=1.8),
    opacity=0.6
))

fig.add_trace(go.Scatter(
    x=train_df["Tanggal"],
    y=train_df["y_pred"],
    mode="lines",
    name="Prediksi (Training)",
    visible=is_visible("Prediksi (Training)"),
    line=dict(color="lightskyblue", width=1.5, dash="dot"),
    opacity=0.7
))

# TESTING
fig.add_trace(go.Scatter(
    x=test_df["Tanggal"],
    y=test_df["y_actual"],
    mode="lines",
    name="Data Aktual (Testing)",
    visible=is_visible("Data Aktual (Testing)"),
    line=dict(color="green", width=2),
))

fig.add_trace(go.Scatter(
    x=test_df["Tanggal"],
    y=test_df["y_pred"],
    mode="lines",
    name="Prediksi (Testing)",
    visible=is_visible("Prediksi (Testing)"),
    line=dict(color="orange", width=2),
))

# =========================
# SPLIT LINE TRAIN → TEST
# =========================
split_date = train_df["Tanggal"].iloc[-1]

fig.add_shape(
    type="line",
    x0=split_date,
    x1=split_date,
    y0=0,
    y1=1,
    yref="paper",
    line=dict(color="gray", width=2, dash="dash"),
)

fig.add_annotation(
    x=split_date,
    y=1.02,
    xref="x",
    yref="paper",
    text="Batas Data Training & Testing",
    showarrow=False,
    font=dict(size=11, color="gray")
)

# =========================
# LAYOUT
# =========================
fig.update_layout(
    xaxis_title="Tanggal",
    yaxis_title="Tinggi Muka Air (cm)",
    height=550,
    hovermode="x unified",
    margin=dict(l=70, r=40, t=90, b=60),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.15,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig, use_container_width=True)


# Model information
with st.expander("🤖 Informasi Model Prediksi", expanded=False):
    col1, col2 = st.columns(2)
    
    # ===== KIRI: PERFORMA MODEL =====
    with col1:
        st.markdown("### **Performa Model**")
        
        col_train, col_test = st.columns(2)
        
        # --- TRAINING ---
        with col_train:
            st.markdown("#### 🏋️ Training")
            st.metric("RMSE", "11.400 cm")
            st.metric("MAE", "3.104 cm")
            st.metric("MAPE", "0.010")
            st.metric("R²", "0.934")
        
        # --- TESTING ---
        with col_test:
            st.markdown("#### 🧪 Testing")
            st.metric("RMSE", "14.860 cm")
            st.metric("MAE", "2.348 cm")
            st.metric("MAPE", "0.0068")
            st.metric("R²", "0.290")
    
    # ===== KANAN: PARAMETER MODEL =====
    with col2:
        st.markdown("### **Parameter Model XGBoost**")
        st.code("""
objective: reg:squarederror
random_state: 42

learning_rate = 0.1
n_estimators  = 50
max_depth    = 7
subsample    = 0.8
colsample_bytree = 0.8
        """, language="python")
        st.markdown("### **Keunggulan Model**")
        st.markdown("""
        - **Lebih unggul** dibandingkan LSTM dan SARIMAX  
        - **Akurasi tinggi** pada data pelatihan  
        - **Efisien secara komputasi** dan cocok untuk sistem operasional
        """)


st.markdown("""
---
**💡 Catatan Prediksi:**
- **Training Phase**: Model dilatih menggunakan 95% data historis (warna biru)
- **Testing Phase**: Testing model menggunakan 5% data terakhir (warna hijau/orange) dengan confidence interval 95%
""")

st.markdown("---")

# Create plotly map with hover details
fig = px.scatter_map(
    create_master_station_data(),
    lat='latitude',
    lon='longitude',
    hover_data={
        'current_level': ':.1f',
        'weather': True,
        'status': True,
    },
    hover_name='name',
    color='status',
    color_discrete_map={
        'Siaga 1': 'red',
        'Siaga 2': 'orange', 
        'Siaga 3': 'yellow',
        'Normal': 'green'
    },
    text='name',
    size_max=25,
    zoom=8,
    height=600
)
fig.update_traces(textposition="top center")
fig.update_layout(
    geo=dict(
        center=dict(lat=-6.2, lon=106.8),
        projection_type="natural earth"
    )
)

config = {
    'scrollZoom': True,
    'doubleClick': 'reset',
    'displayModeBar': True,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
}

st.plotly_chart(fig, use_container_width=True, config=config)

# Header
st.markdown(f"## Status Tinggi Muka Air Saat Ini")
st.markdown(f"**🕐 Last Updated:** {last_updated}")

# Main status display
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    # Katulampa - UPSTREAM
    if current_katulampa is not None:
        alert_status, alert_type = get_alert_level('Katulampa', current_katulampa)
        threshold_info = get_threshold_info('Katulampa')
        
        st.markdown("### Katulampa")
        st.markdown("*⬆️ Hulu Sungai*")
        st.metric(
            label="Tinggi Muka Air",
            value=f"{current_katulampa:.1f} cm",
            delta=f"{delta_katulampa:+.1f} cm",
            delta_color="inverse"
        )
        
        # Alert status - NO dropdown here
        if alert_type == "error":
            st.error(alert_status)
        elif alert_type == "warning":
            st.warning(alert_status)
        elif alert_type == "info":
            st.info(alert_status)
        else:
            st.success(alert_status)
        
        # Weather
        st.markdown(f"🌤️ **Cuaca:** {weather_katulampa}")
        
        # Separate info dropdown below weather
        with st.expander("ℹ️ Informasi Ambang Batas Siaga"):
            st.markdown(f"""
            **{threshold_info['name']} - Ambang Batas Siaga:**
            
            🟢 **Normal**: < {threshold_info['siaga3']}
            
            🟡 **Siaga 3**: {threshold_info['siaga3']}
            
            🟠 **Siaga 2**: {threshold_info['siaga2']}
            
            🔴 **Siaga 1**: {threshold_info['siaga1']}                    
        """)
    else:
        st.markdown("### Katulampa")
        st.markdown("*⬆️ Hulu Sungai*")
        st.error("Data tidak tersedia")

with col2:
    # Depok
    if current_depok is not None:
        alert_status, alert_type = get_alert_level('Depok', current_depok)
        threshold_info = get_threshold_info('Depok')
        
        st.markdown("### Depok")
        st.markdown("*↕️ Tengah Aliran*")
        st.metric(
            label="Tinggi Muka Air",
            value=f"{current_depok:.1f} cm",
            delta=f"{delta_depok:+.1f} cm",
            delta_color="inverse"
        )
        
        # Alert status - NO dropdown here
        if alert_type == "error":
            st.error(alert_status)
        elif alert_type == "warning":
            st.warning(alert_status)
        elif alert_type == "info":
            st.info(alert_status)
        else:
            st.success(alert_status)
        
        # Weather
        st.markdown(f"🌤️ **Cuaca:** {weather_depok}")
        
        # Separate info dropdown below weather
        with st.expander("ℹ️ Informasi Ambang Batas Siaga"):
            st.markdown(f"""
            **{threshold_info['name']} - Ambang Batas Siaga:**
            
            🟢 **Normal**: < {threshold_info['siaga3']}
            
            🟡 **Siaga 3**: {threshold_info['siaga3']}
            
            🟠 **Siaga 2**: {threshold_info['siaga2']}
            
            🔴 **Siaga 1**: {threshold_info['siaga1']}                    
        """)
    else:
        st.markdown("### Depok")
        st.markdown("*↕️ Tengah Aliran*")
        st.error("Data tidak tersedia")

with col3:
    # Manggarai - MAIN FOCUS
    if current_manggarai is not None:
        alert_status, alert_type = get_alert_level('Manggarai', current_manggarai)
        threshold_info = get_threshold_info('Manggarai')
        
        st.markdown("### **MANGGARAI**")
        st.markdown("*⬇️ Pintu Air Aliran Sungai Ciliwung di Pusat Kota Jakarta*")
        st.metric(
            label="Tinggi Muka Air",
            value=f"{current_manggarai:.1f} cm",
            delta=f"{delta_manggarai:+.1f} cm",
            delta_color="inverse"
        )
        
        # Alert status - NO dropdown here
        if alert_type == "error":
            st.error(alert_status)
        elif alert_type == "warning":
            st.warning(alert_status)
        elif alert_type == "info":
            st.info(alert_status)
        else:
            st.success(alert_status)
        
        # Weather
        st.markdown(f"🌤️ **Cuaca:** {weather_manggarai}")
        
        # Separate info dropdown below weather
        with st.expander("ℹ️ Informasi Ambang Batas Siaga"):
            st.markdown(f"""
            **{threshold_info['name']} - Ambang Batas Siaga:**
            
            🟢 **Normal**: < {threshold_info['siaga3']}
            
            🟡 **Siaga 3**: {threshold_info['siaga3']}
            
            🟠 **Siaga 2**: {threshold_info['siaga2']}
            
            🔴 **Siaga 1**: {threshold_info['siaga1']}                    
        """)
    else:
        st.markdown("### **MANGGARAI**")
        st.error("Data tidak tersedia")


show_footer()
