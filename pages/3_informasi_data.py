import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import seasonal_decompose
from utils.data_loader import load_main_data
from utils.footer import show_footer

st.title("📊 Informasi dan Analisis Data")

# Load data
try:
    df_final = load_main_data()
    # Set datetime index if not already set
    if not isinstance(df_final.index, pd.DatetimeIndex):
        if 'Tanggal' in df_final.columns:
            df_final['Tanggal'] = pd.to_datetime(df_final['Tanggal'])
            df_final.set_index('Tanggal', inplace=True)
        elif df_final.index.name == 'Tanggal' or 'Tanggal' in str(df_final.index.name):
            df_final.index = pd.to_datetime(df_final.index)
    
    st.success(f"Data berhasil dimuat: {len(df_final)} baris, {len(df_final.columns)} kolom")
    
    # Show basic info
    st.subheader("📋 Informasi Dasar Dataset")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Baris", len(df_final))
    with col2:
        st.metric("Total Kolom", len(df_final.columns))
    with col3:
        st.metric("Periode Awal", df_final.index.min().strftime('%Y-%m-%d'))
    with col4:
        st.metric("Periode Akhir", df_final.index.max().strftime('%Y-%m-%d'))
    
    # Get numeric columns (air level data)
    numeric_columns = [col for col in df_final.columns if '(air)' in col]
    clean_labels = [col.replace(' (air)', '') for col in numeric_columns]
    
    # === CORRELATION ANALYSIS ===
    st.subheader("🔗 Analisis Korelasi")
    
    # Calculate correlation matrix
    corr = df_final[numeric_columns].corr()
    
    # Create interactive heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=clean_labels,
        y=clean_labels,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr.values, 3),
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Matriks Korelasi Tinggi Muka Air",
        xaxis_title="Stasiun",
        yaxis_title="Stasiun",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # === TIME SERIES PLOTS ===
    st.subheader("📈 Plot Time Series Interaktif")
    
    # Create interactive time series plot
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    for i, col in enumerate(numeric_columns):
        fig.add_trace(go.Scatter(
            x=df_final.index,
            y=df_final[col],
            mode='lines',
            name=col.replace(' (air)', ''),
            line=dict(color=colors[i % len(colors)], width=2),
            opacity=0.8
        ))
    
    fig.update_layout(
        title="Time Series Tinggi Muka Air - Semua Stasiun",
        xaxis_title="Tanggal",
        yaxis_title="Tinggi Muka Air (cm)",
        height=600,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual station analysis
    st.subheader("📊 Analisis per Stasiun")
    selected_station = st.selectbox(
        "Pilih stasiun untuk analisis detail:",
        options=clean_labels,
        index=clean_labels.index('Manggarai') if 'Manggarai' in clean_labels else 0
    )
    
    selected_col = f"{selected_station} (air)"
    
    # Create subplot for selected station
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Time Series', 'Distribusi (Histogram)', 
                       'Boxplot', 'Trend Bulanan'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Time series
    fig.add_trace(
        go.Scatter(x=df_final.index, y=df_final[selected_col], 
                  mode='lines', name='Tinggi Muka Air',
                  line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=df_final[selected_col], nbinsx=50, 
                    name='Distribusi', marker_color='lightblue'),
        row=1, col=2
    )
    
    # Boxplot
    fig.add_trace(
        go.Box(y=df_final[selected_col], name='Boxplot', 
               marker_color='lightgreen'),
        row=2, col=1
    )
    
    # Monthly trend
    monthly_avg = df_final[selected_col].resample('M').mean()
    fig.add_trace(
        go.Scatter(x=monthly_avg.index, y=monthly_avg.values,
                  mode='lines+markers', name='Rata-rata Bulanan',
                  line=dict(color='red', width=3)),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f"Analisis Detail - Stasiun {selected_station}",
        height=800,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # === CROSS CORRELATION ANALYSIS ===
    st.subheader("🔄 Analisis Cross-Correlation dengan Manggarai")
    
    if 'Manggarai (air)' in df_final.columns:
        max_lag = st.slider("Maksimum Lag (jam):", min_value=12, max_value=48, value=24)
        
        # Calculate cross-correlation
        target_stations = ["Katulampa (air)", "Depok (air)", "Manggarai (air)"]
        available_stations = [col for col in target_stations if col in df_final.columns]
        
        fig = go.Figure()
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        summary_data = []
        
        for i, col in enumerate(available_stations):
            lags = list(range(1, max_lag+1))
            corrs = []
            
            for lag in lags:
                shifted = df_final[col].shift(lag)
                corr = shifted.corr(df_final['Manggarai (air)'])
                corrs.append(corr)
            
            color = colors[i % len(colors)]
            station_name = col.replace(" (air)", "")
            
            # Add line
            fig.add_trace(go.Scatter(
                x=lags, y=corrs,
                mode='lines+markers',
                name=station_name,
                line=dict(color=color, width=3),
                marker=dict(size=6)
            ))
            
            # Find and mark maximum
            max_idx = np.argmax(corrs)
            max_lag_val = lags[max_idx]
            max_corr_val = corrs[max_idx]
            
            fig.add_trace(go.Scatter(
                x=[max_lag_val], y=[max_corr_val],
                mode='markers',
                name=f'{station_name} Max',
                marker=dict(color=color, size=15, symbol='star'),
                showlegend=False
            ))
            
            # Add to summary
            summary_data.append({
                'Stasiun': station_name,
                'Lag Optimal (jam)': max_lag_val,
                'Korelasi Maksimum': f"{max_corr_val:.4f}",
                'Korelasi Lag-1': f"{corrs[0]:.4f}" if len(corrs) > 0 else "N/A"
            })
        
        fig.update_layout(
            title="Cross-Correlation dengan Manggarai (air)",
            xaxis_title="Lag (jam)",
            yaxis_title="Korelasi",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary table
        st.subheader("📊 Ringkasan Cross-Correlation")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    # === SEASONAL DECOMPOSITION ===
    st.subheader("🔄 Dekomposisi Time Series")
    
    if selected_col in df_final.columns:
        period = st.selectbox("Pilih periode untuk dekomposisi:", 
                             options=[24, 168], 
                             format_func=lambda x: f"{x} jam ({'Harian' if x==24 else 'Mingguan'})",
                             index=0)
        
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                df_final[selected_col].dropna(),
                model='additive',
                period=period,
                extrapolate_trend='freq'
            )
            
            # Create interactive decomposition plot
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Data Asli', 'Trend', 'Seasonal', 'Residual'),
                vertical_spacing=0.05
            )
            
            # Original
            fig.add_trace(
                go.Scatter(x=decomposition.observed.index, 
                          y=decomposition.observed.values,
                          mode='lines', name='Original',
                          line=dict(color='blue', width=1)),
                row=1, col=1
            )
            
            # Trend
            fig.add_trace(
                go.Scatter(x=decomposition.trend.index, 
                          y=decomposition.trend.values,
                          mode='lines', name='Trend',
                          line=dict(color='red', width=2)),
                row=2, col=1
            )
            
            # Seasonal
            fig.add_trace(
                go.Scatter(x=decomposition.seasonal.index, 
                          y=decomposition.seasonal.values,
                          mode='lines', name='Seasonal',
                          line=dict(color='green', width=1)),
                row=3, col=1
            )
            
            # Residual
            fig.add_trace(
                go.Scatter(x=decomposition.resid.index, 
                          y=decomposition.resid.values,
                          mode='lines', name='Residual',
                          line=dict(color='purple', width=1)),
                row=4, col=1
            )
            
            fig.update_layout(
                title=f"Dekomposisi Time Series - {selected_station} (periode {period} jam)",
                height=800,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Seasonal pattern detail
            st.subheader("🔍 Pola Seasonal Detail")
            
            # Show one complete cycle
            seasonal_data = decomposition.seasonal.head(period * 5)  # 5 cycles
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=seasonal_data.index,
                y=seasonal_data.values,
                mode='lines',
                name='Pola Seasonal',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title=f"Pola Seasonal {selected_station} (5 siklus pertama)",
                xaxis_title="Waktu",
                yaxis_title="Komponen Seasonal",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error dalam dekomposisi: {str(e)}")
            st.info("Coba dengan periode yang berbeda atau pastikan data memiliki observasi yang cukup.")
    
    # === DATA STATISTICS ===
    st.subheader("📈 Statistik Deskriptif")
    
    stats_df = df_final[numeric_columns].describe()
    st.dataframe(stats_df.round(2), use_container_width=True)
    
    # === MISSING VALUES ANALYSIS ===
    st.subheader("❓ Analisis Missing Values")
    
    missing_data = df_final.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        fig = go.Figure(data=[
            go.Bar(x=missing_data.index, y=missing_data.values,
                   marker_color='lightcoral')
        ])
        
        fig.update_layout(
            title="Missing Values per Kolom",
            xaxis_title="Kolom",
            yaxis_title="Jumlah Missing Values",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(missing_data.to_frame('Missing Values'), use_container_width=True)
    else:
        st.success("✅ Tidak ada missing values dalam dataset!")

except FileNotFoundError:
    st.error("❌ File '02 All Data.csv' tidak ditemukan di folder 'data/'. Pastikan file tersebut tersedia.")
except Exception as e:
    st.error(f"❌ Error saat memuat data: {str(e)}")

# === INSIGHTS AND INTERPRETATION ===
st.subheader("💡 Insights dan Interpretasi")

with st.expander("📖 Penjelasan Analisis", expanded=False):
    st.markdown("""
    ### 🔗 Analisis Korelasi
    - **Warna merah**: Korelasi positif kuat
    - **Warna biru**: Korelasi negatif 
    - **Hover** pada cell untuk melihat nilai eksak
    
    ### 📈 Time Series Interaktif
    - **Zoom**: Drag untuk zoom area tertentu
    - **Pan**: Shift+drag untuk menggeser
    - **Toggle**: Klik legend untuk hide/show stasiun
    
    ### 🔄 Cross-Correlation
    - **Bintang**: Menandai lag optimal
    - **Hover**: Lihat nilai korelasi detail
    - Lag tinggi normal untuk stasiun hulu (Katulampa)
    
    ### 📊 Dekomposisi
    - **Trend**: Pola jangka panjang
    - **Seasonal**: Pola berulang harian/mingguan
    - **Residual**: Noise setelah trend dan seasonal dihilangkan
    
    ### 💡 Tips Navigasi
    - Semua chart bisa di-zoom dan pan
    - Double-click untuk reset view
    - Download chart dengan klik camera icon
    """)

show_footer()