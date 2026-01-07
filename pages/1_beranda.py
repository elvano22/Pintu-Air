# import streamlit as st
# from utils.data_loader import load_main_data, load_info_data, create_master_station_data, get_station_data
# import pandas as pd
# import numpy as np
# import base64
# import plotly.express as px
# from utils.footer import show_footer
# import xgboost as xgb
# import numpy as np
# from sklearn.metrics import mean_squared_error, r2_score
# import plotly.graph_objects as go
# import io

# # Cache
# data = load_main_data()

# with open("assets/images/logo_full.svg", "rb") as img_file:
#     logo_base64 = base64.b64encode(img_file.read()).decode()

# st.markdown(f"""
# <div style="text-align: center; padding: 0 !important">
#     <img src="data:image/svg+xml;base64,{logo_base64}" width="400">
#     <h1>Pintu Air Manggarai</h1>
# </div>
# """, unsafe_allow_html=True)
# st.markdown("Website ini dikhususkan untuk **monitoring dan prediksi** tinggi muka air di **Pintu Air Manggarai**. Data dari Katulampa dan Depok ditampilkan sebagai **indikator upstream** yang mempengaruhi kondisi di Manggarai.")
# st.markdown("**Pintu Air Manggarai** berada pada aliran **Sungai Ciliwung**. Terdapat beberapa pintu air di sepanjang aliran Sungai Ciliwung. Dimulai dari paling hulu sungai, yaitu Pintu Air Katulampa -> Pintu Air Depok -> **Pintu Air Manggarai**")

# def get_alert_level(location, height):
#     station = get_station_data(location)
    
#     if height >= station['siaga_1']:
#         return "🔴 Siaga 1", "error"
#     elif height >= station['siaga_2']:
#         return "🟠 Siaga 2", "warning" 
#     elif height >= station['siaga_3']:
#         return "🟡 Siaga 3", "info"
#     else:
#         return "🟢 Normal", "success"

# def load_prediction_model():
#     """Load the trained XGBoost model"""
#     model = xgb.XGBRegressor()
#     model.load_model('models/13_best_model.json')
    
#     return model

# def predict_6_hours_ahead(model, data):
#     """Predict 6 hours ahead based on the last data point"""
#     feature_columns = [
#         # Time features
#         'hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos',
#         # Manggarai lag features
#         'manggarai_air_lag1', 'manggarai_air_lag2', 'manggarai_air_lag3', 
#         'manggarai_air_lag4', 'manggarai_air_lag5', 'manggarai_air_lag6',
#         'manggarai_air_lag7', 'manggarai_air_lag9', 'manggarai_air_lag12',
#         'manggarai_air_lag14', 'manggarai_air_lag15', 'manggarai_air_lag17',
#         'manggarai_air_lag20', 'manggarai_air_lag21',
#         # Other features from the existing feature space
#     ]
    
#     # Get the last row for prediction
#     last_row = data.iloc[-1]
    
#     # Prepare features for last time point (you might need to dynamically compute these)
#     current_features = [
#         np.sin(2 * np.pi * last_row.name.hour / 24),  # hour_sin
#         np.cos(2 * np.pi * last_row.name.hour / 24),  # hour_cos
#         np.sin(2 * np.pi * last_row.name.dayofyear / 365),  # dayofyear_sin
#         np.cos(2 * np.pi * last_row.name.dayofyear / 365),  # dayofyear_cos
#     ]
    
#     # Add lag features (assuming they are available)
#     for col in feature_columns[4:]:
#         if col in data.columns:
#             current_features.append(last_row.get(col, 0))
#         else:
#             current_features.append(0)
    
#     # Predict
#     prediction = model.predict([current_features])[0]
    
#     return max(0, prediction)  # Ensure non-negative prediction

# def create_short_forecast(pred_data):
#     # Code for short forecast visualization remains the same as in your original script
#     # This is just a placeholder - you'll copy the existing implementation
#     pass

# # File Upload Section
# st.sidebar.header("Upload CSV for Prediction")
# uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:
#     # Read the uploaded CSV
#     df = pd.read_csv(uploaded_file)
    
#     # Load model
#     model = load_prediction_model()
    
#     # Predict
#     feature_columns = [
#         'hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos',
#         # Add all other lag features from your original feature space
#     ]
    
#     # Perform prediction
#     predictions = model.predict(df[feature_columns])
#     df['predicted_manggarai_air'] = predictions
    
#     # Create download button
#     csv = df.to_csv(index=False)
#     st.sidebar.download_button(
#         label="Download predicted CSV",
#         data=csv,
#         file_name='predicted_data.csv',
#         mime='text/csv',
#     )

# # Existing Dashboard Code
# @st.cache_resource
# def load_model_and_predict():
#     model = load_prediction_model()
#     current_data = data.copy()
    
#     # Prediction for dashboard
#     six_hour_prediction = predict_6_hours_ahead(model, current_data)
    
#     return model, six_hour_prediction

# # Main dashboard prediction
# model, six_hour_prediction = load_model_and_predict()

# # Existing dashboard visualizations and code continue here...
# st.subheader("🎯 Prediksi Jangka Pendek (6 Jam Ke Depan)")
# st.write(f"Prediksi tinggi muka air dalam 6 jam: **{six_hour_prediction:.2f} cm**")


# import streamlit as st
# from utils.data_loader import load_main_data, load_info_data, create_master_station_data, get_station_data
# import pandas as pd
# import numpy as np
# import base64
# import plotly.express as px
# from utils.footer import show_footer
# import xgboost as xgb
# import numpy as np
# from sklearn.metrics import mean_squared_error, r2_score
# import plotly.graph_objects as go

# # Cache


# with open("assets/images/logo_full.svg", "rb") as img_file:
#     logo_base64 = base64.b64encode(img_file.read()).decode()

# st.markdown(f"""
# <div style="text-align: center; padding: 0 !important">
#     <img src="data:image/svg+xml;base64,{logo_base64}" width="400">
#     <h1>Pintu Air Manggarai</h1>
# </div>
# """, unsafe_allow_html=True)
# st.markdown("Website ini dikhususkan untuk **monitoring dan prediksi** tinggi muka air di **Pintu Air Manggarai**. Data dari Katulampa dan Depok ditampilkan sebagai **indikator upstream** yang mempengaruhi kondisi di Manggarai.")
# st.markdown("**Pintu Air Manggarai** berada pada aliran **Sungai Ciliwung**. Terdapat beberapa pintu air di sepanjang aliran Sungai Ciliwung. Dimulai dari paling hulu sungai, yaitu Pintu Air Katulampa -> Pintu Air Depok -> **Pintu Air Manggarai**")

# def get_alert_level(location, height):
#     station = get_station_data(location)
    
#     if height >= station['siaga_1']:
#         return "🔴 Siaga 1", "error"
#     elif height >= station['siaga_2']:
#         return "🟠 Siaga 2", "warning" 
#     elif height >= station['siaga_3']:
#         return "🟡 Siaga 3", "info"
#     else:
#         return "🟢 Normal", "success"
    
# # Function to get threshold info for each location

    





# # === PREDICTION SECTION ===
# st.subheader("📈 Prediksi Tinggi Muka Air Manggarai")

# # Load the trained model and data
# @st.cache_resource
# def load_prediction_model():
#     """Load the trained XGBoost model"""
#     try:
#         # Load the saved model
#         model = xgb.XGBRegressor()
#         model.load_model('models/13_best_model.json')
        
#         # Model parameters for display
#         model_params = {
#             'colsample_bytree': 0.8, 
#             'learning_rate': 0.05, 
#             'max_depth': 4, 
#             'min_child_weight': 3, 
#             'n_estimators': 300, 
#             'reg_alpha': 0.5, 
#             'reg_lambda': 1.0, 
#             'subsample': 0.8,
#             'early_stopping_rounds': 50
#         }
        
#         return model, model_params
            
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         st.info("Pastikan file 'models/13_best_model.json' tersedia")
#         return None, {}

# @st.cache_data
# def create_prediction_data():
#     """Create prediction data using the loaded model"""
    
#     # Load the main data
#     df = load_main_data()
    
#     # Ensure datetime index
#     if 'Tanggal' in df.columns:
#         df['Tanggal'] = pd.to_datetime(df['Tanggal'])
#         df.set_index('Tanggal', inplace=True)
#     elif not isinstance(df.index, pd.DatetimeIndex):
#         df.index = pd.to_datetime(df.index)
    
#     # Load model
#     model, _ = load_prediction_model()
    
#     if model is None:
#         # Fallback to synthetic data if model can't be loaded
#         return create_synthetic_predictions(df)
    
#     # Create the EXACT features that the model expects
#     manggarai_data = df['Manggarai (air)'].dropna()
#     depok_data = df['Depok (air)'].dropna() if 'Depok (air)' in df.columns else pd.Series()
#     katulampa_data = df['Katulampa (air)'].dropna() if 'Katulampa (air)' in df.columns else pd.Series()
    
#     # Create features dataframe with exact column names that model expects
#     features_df = pd.DataFrame(index=manggarai_data.index)
    
#     # Time features
#     features_df['hour_sin'] = np.sin(2 * np.pi * features_df.index.hour / 24)
#     features_df['hour_cos'] = np.cos(2 * np.pi * features_df.index.hour / 24)
    
#     # Manggarai air lag features
#     for lag in range(1, 7):  # lag 1-6
#         features_df[f'manggarai_air_lag{lag}'] = manggarai_data.shift(lag)
    
#     # Depok air lag features
#     if len(depok_data) > 0:
#         for lag in [6, 7, 8, 9]:
#             features_df[f'depok_air_lag{lag}'] = depok_data.reindex(features_df.index).shift(lag)
#     else:
#         # Fill with zeros if no Depok data
#         for lag in [6, 7, 8, 9]:
#             features_df[f'depok_air_lag{lag}'] = 0
    
#     # Katulampa air lag features
#     if len(katulampa_data) > 0:
#         for lag in [11, 12, 13]:
#             features_df[f'katulampa_air_lag{lag}'] = katulampa_data.reindex(features_df.index).shift(lag)
#     else:
#         # Fill with zeros if no Katulampa data
#         for lag in [11, 12, 13]:
#             features_df[f'katulampa_air_lag{lag}'] = 0
    
#     # Weather features (binary: 1 for 'hujan', 0 for others)
#     def create_weather_lag_features(location, data_col, lag_list):
#         if data_col in df.columns:
#             weather_data = df[data_col].fillna('cerah')
#             weather_binary = (weather_data == 'hujan').astype(int)
#             for lag in lag_list:
#                 features_df[f'{location}_cuaca_lag{lag}_hujan'] = weather_binary.reindex(features_df.index).shift(lag)
#         else:
#             # Fill with zeros if no weather data
#             for lag in lag_list:
#                 features_df[f'{location}_cuaca_lag{lag}_hujan'] = 0
    
#     # Create weather lag features
#     create_weather_lag_features('manggarai', 'Manggarai (cuaca)', [1, 2, 3, 4, 5, 6])
#     create_weather_lag_features('depok', 'Depok (cuaca)', [6, 7, 8, 9])
#     create_weather_lag_features('katulampa', 'Katulampa (cuaca)', [11, 12, 13])
    
#     # Drop rows with NaN values (due to lag features)
#     features_df = features_df.dropna()
#     target_data = manggarai_data.loc[features_df.index]
    
#     # Split data (80% train, 20% test)
#     split_idx = int(len(features_df) * 0.8)
    
#     X_train = features_df.iloc[:split_idx]
#     X_test = features_df.iloc[split_idx:]
#     y_train = target_data.iloc[:split_idx]
#     y_test = target_data.iloc[split_idx:]
    
#     try:
#         # Make predictions using the loaded model
#         train_pred = model.predict(X_train)
#         test_pred = model.predict(X_test)
        
#         # Create future features for forecasting (next 72 hours)
#         forecast_predictions = []
#         forecast_dates = []
        
#         # Start from the last available features
#         last_features = features_df.tail(1).copy()
#         last_date = features_df.index[-1]
        
#         # Get recent data for upstream predictions (simple approach)
#         recent_manggarai = manggarai_data.tail(20).values
#         recent_depok = depok_data.reindex(manggarai_data.index).tail(20).values if len(depok_data) > 0 else np.full(20, manggarai_data.tail(20).mean())
#         recent_katulampa = katulampa_data.reindex(manggarai_data.index).tail(20).values if len(katulampa_data) > 0 else np.full(20, manggarai_data.tail(20).mean())
        
#         # Simple forecasting with trend continuation for upstream stations
#         for i in range(72):  # 72 hours forecast
#             # Get current features
#             current_features = last_features.iloc[0].copy()
            
#             # Update time features for next hour
#             next_date = last_date + pd.Timedelta(hours=i+1)
#             current_features['hour_sin'] = np.sin(2 * np.pi * next_date.hour / 24)
#             current_features['hour_cos'] = np.cos(2 * np.pi * next_date.hour / 24)
            
#             # For upstream stations, use trend continuation + seasonal pattern
#             # Depok forecast (simple seasonal + trend)
#             if i < len(recent_depok) - 9:
#                 # Use actual recent data
#                 for lag in [6, 7, 8, 9]:
#                     if lag - 1 < len(recent_depok):
#                         current_features[f'depok_air_lag{lag}'] = recent_depok[-(lag-i)] if (lag-i) > 0 else recent_depok[-1]
#             else:
#                 # Use trend continuation for Depok
#                 depok_trend = np.mean(recent_depok[-5:]) if len(recent_depok) >= 5 else np.mean(recent_depok)
#                 seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * next_date.hour / 24)  # Daily seasonality
#                 for lag in [6, 7, 8, 9]:
#                     current_features[f'depok_air_lag{lag}'] = depok_trend * seasonal_factor
            
#             # Katulampa forecast (simple seasonal + trend)
#             if i < len(recent_katulampa) - 13:
#                 # Use actual recent data
#                 for lag in [11, 12, 13]:
#                     if lag - 1 < len(recent_katulampa):
#                         current_features[f'katulampa_air_lag{lag}'] = recent_katulampa[-(lag-i)] if (lag-i) > 0 else recent_katulampa[-1]
#             else:
#                 # Use trend continuation for Katulampa
#                 katulampa_trend = np.mean(recent_katulampa[-5:]) if len(recent_katulampa) >= 5 else np.mean(recent_katulampa)
#                 seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * next_date.hour / 24)  # Daily seasonality
#                 for lag in [11, 12, 13]:
#                     current_features[f'katulampa_air_lag{lag}'] = katulampa_trend * seasonal_factor
            
#             # Weather features - assume no rain for simplicity (could be improved with weather forecast API)
#             weather_locations = ['manggarai', 'depok', 'katulampa']
#             weather_lags = {'manggarai': [1,2,3,4,5,6], 'depok': [6,7,8,9], 'katulampa': [11,12,13]}
            
#             for location in weather_locations:
#                 for lag in weather_lags[location]:
#                     current_features[f'{location}_cuaca_lag{lag}_hujan'] = 0  # Assume no rain
            
#             # Predict next value
#             try:
#                 next_pred = model.predict([current_features.values])[0]
#                 next_pred = max(0, next_pred)  # Ensure non-negative
#                 forecast_predictions.append(next_pred)
#                 forecast_dates.append(next_date)
                
#                 # Update Manggarai lag features for next iteration
#                 # Shift manggarai lags
#                 for lag in range(6, 1, -1):
#                     if f'manggarai_air_lag{lag}' in current_features.index:
#                         current_features[f'manggarai_air_lag{lag}'] = current_features[f'manggarai_air_lag{lag-1}']
                
#                 current_features['manggarai_air_lag1'] = next_pred
                
#                 # Update last_features for next iteration
#                 last_features.iloc[0] = current_features
                
#             except Exception as e:
#                 # If prediction fails, break the loop
#                 st.warning(f"Forecast stopped at hour {i+1} due to: {str(e)}")
#                 break
        
#         forecast_pred = pd.Series(forecast_predictions, index=forecast_dates)
        
#         # Calculate confidence intervals
#         if len(test_pred) > 0:
#             test_residuals = y_test - test_pred
#             residual_std = np.std(test_residuals)
#         else:
#             residual_std = manggarai_data.std() * 0.1  # Fallback
        
#         test_upper = pd.Series(test_pred, index=y_test.index) + 1.96 * residual_std
#         test_lower = pd.Series(test_pred, index=y_test.index) - 1.96 * residual_std
        
#         forecast_upper = forecast_pred + 1.96 * residual_std * 1.2
#         forecast_lower = forecast_pred - 1.96 * residual_std * 1.2
        
#         return {
#             'train_actual': y_train,
#             'train_pred': pd.Series(train_pred, index=y_train.index),
#             'test_actual': y_test,
#             'test_pred': pd.Series(test_pred, index=y_test.index),
#             'test_upper': test_upper,
#             'test_lower': test_lower,
#             'forecast_pred': forecast_pred,
#             'forecast_upper': forecast_upper,
#             'forecast_lower': forecast_lower
#         }
        
#     except Exception as e:
#         st.error(f"Error making predictions: {str(e)}")
#         return create_synthetic_predictions(df)

# def create_synthetic_predictions(df):
#     """Fallback synthetic predictions if model loading fails"""
#     manggarai_data = df['Manggarai (air)'].dropna()
    
#     # Split data (80% train, 20% test)
#     split_idx = int(len(manggarai_data) * 0.8)
#     train_data = manggarai_data.iloc[:split_idx]
#     test_data = manggarai_data.iloc[split_idx:]
    
#     # Create synthetic predictions for demo
#     np.random.seed(42)
#     train_pred = train_data + np.random.normal(0, 2, len(train_data))
#     test_pred = test_data + np.random.normal(0, 3, len(test_data))
    
#     # Future forecast (next 72 hours)
#     last_date = manggarai_data.index[-1]
#     future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), 
#                                 periods=72, freq='h')
    
#     recent_mean = manggarai_data.tail(168).mean()
#     recent_std = manggarai_data.tail(168).std()
    
#     forecast_values = []
#     for i, date in enumerate(future_dates):
#         daily_pattern = 5 * np.sin(2 * np.pi * date.hour / 24)
#         weekly_pattern = 2 * np.sin(2 * np.pi * date.dayofweek / 7)
#         noise = np.random.normal(0, recent_std * 0.3)
#         forecast_val = recent_mean + daily_pattern + weekly_pattern + noise
#         forecast_values.append(max(0, forecast_val))
    
#     forecast_pred = pd.Series(forecast_values, index=future_dates)
    
#     # Calculate confidence intervals
#     test_residuals = test_data - test_pred
#     residual_std = np.std(test_residuals)
    
#     test_upper = test_pred + 1.96 * residual_std
#     test_lower = test_pred - 1.96 * residual_std
    
#     forecast_upper = forecast_pred + 1.96 * residual_std * 1.2
#     forecast_lower = forecast_pred - 1.96 * residual_std * 1.2
    
#     return {
#         'train_actual': train_data,
#         'train_pred': train_pred,
#         'test_actual': test_data,
#         'test_pred': test_pred,
#         'test_upper': test_upper,
#         'test_lower': test_lower,
#         'forecast_pred': forecast_pred,
#         'forecast_upper': forecast_upper,
#         'forecast_lower': forecast_lower
#     }

# # Load model and data
# model, model_params = load_prediction_model()
# pred_data = create_prediction_data()

# # Calculate metrics
# if len(pred_data['test_actual']) > 0:
#     test_rmse = np.sqrt(mean_squared_error(pred_data['test_actual'], pred_data['test_pred']))
#     test_r2 = r2_score(pred_data['test_actual'], pred_data['test_pred'])
# else:
#     test_rmse, test_r2 = 0, 0

# # === SHORT-TERM FORECAST (6 HOURS) ===
# st.subheader("🎯 Prediksi Jangka Pendek (6 Jam Ke Depan)")

# # Create short-term forecast table and plot
# if len(pred_data['forecast_pred']) >= 6:
#     # Get first 6 hours of forecast
#     short_forecast = pred_data['forecast_pred'].head(6)
#     short_upper = pred_data['forecast_upper'].head(6)
#     short_lower = pred_data['forecast_lower'].head(6)
    
#     # Create forecast table
#     forecast_table = pd.DataFrame({
#         'Jam Ke-': [f"+{i+1}" for i in range(6)],
#         'Waktu': [dt.strftime('%H:%M') for dt in short_forecast.index],
#         'Tanggal': [dt.strftime('%Y-%m-%d') for dt in short_forecast.index],
#         'Prediksi (cm)': [f"{val:.1f}" for val in short_forecast.values],
#         'Range Min (cm)': [f"{val:.1f}" for val in short_lower.values],
#         'Range Max (cm)': [f"{val:.1f}" for val in short_upper.values],
#     })
    
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.dataframe(forecast_table, use_container_width=True, hide_index=True)
    
#     with col2:
#         # Create short-term forecast plot
#         fig_short = go.Figure()
        
#         # Add current data (last 24 hours)
#         recent_data = pred_data['test_actual'].tail(24) if len(pred_data['test_actual']) >= 24 else pred_data['test_actual']
        
#         fig_short.add_trace(go.Scatter(
#             x=recent_data.index,
#             y=recent_data.values,
#             mode='lines+markers',
#             name='Data Aktual (24 jam terakhir)',
#             line=dict(color='blue', width=2),
#             marker=dict(size=4)
#         ))
        
#         # Add forecast
#         fig_short.add_trace(go.Scatter(
#             x=short_forecast.index,
#             y=short_forecast.values,
#             mode='lines+markers',
#             name='Prediksi 6 Jam',
#             line=dict(color='red', width=3, dash='dash'),
#             marker=dict(size=8, symbol='diamond')
#         ))
        
#         # Add confidence interval
#         fig_short.add_trace(go.Scatter(
#             x=short_upper.index,
#             y=short_upper.values,
#             mode='lines',
#             line=dict(width=0),
#             showlegend=False,
#             hoverinfo='skip'
#         ))
        
#         fig_short.add_trace(go.Scatter(
#             x=short_lower.index,
#             y=short_lower.values,
#             mode='lines',
#             line=dict(width=0),
#             fill='tonexty',
#             fillcolor='rgba(255, 0, 0, 0.2)',
#             name='Confidence Interval (95%)',
#             hoverinfo='skip'
#         ))
        
#         # Add alert level horizontal lines
#         alert_levels = {
#             'Siaga 1': {'value': 950, 'color': 'red', 'dash': 'solid'},
#             'Siaga 2': {'value': 850, 'color': 'orange', 'dash': 'dash'},
#             'Siaga 3': {'value': 750, 'color': 'gold', 'dash': 'dot'}
#         }
        
#         for level_name, level_info in alert_levels.items():
#             fig_short.add_hline(
#                 y=level_info['value'],
#                 line_dash=level_info['dash'],
#                 line_color=level_info['color'],
#                 line_width=2,
#                 annotation_text=f"{level_name} ({level_info['value']} cm)",
#                 annotation_position="right"
#             )
        
#         # Add current time line
#         current_time = recent_data.index[-1] if len(recent_data) > 0 else pd.Timestamp.now()
#         fig_short.add_shape(
#             type="line",
#             x0=current_time, x1=current_time,
#             y0=0, y1=1,
#             yref="paper",
#             line=dict(color="green", width=2, dash="dot"),
#         )
        
#         fig_short.add_annotation(
#             x=current_time,
#             y=1.02,
#             yref="paper",
#             text="Sekarang",
#             showarrow=False,
#             font=dict(size=10, color="green")
#         )
        
#         fig_short.update_layout(
#             title="Prediksi 6 Jam Ke Depan",
#             xaxis_title="Waktu",
#             yaxis_title="Tinggi Muka Air (cm)",
#             height=400,
#             hovermode='x unified'
#         )
        
#         st.plotly_chart(fig_short, use_container_width=True)

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
    model = xgb.XGBRegressor()
    model.load_model('models/13_best_model.json')
    
    model_params = {
        'colsample_bytree': 0.8, 
        'learning_rate': 0.05, 
        'max_depth': 4, 
        'min_child_weight': 3, 
        'n_estimators': 300, 
        'reg_alpha': 0.5, 
        'reg_lambda': 1.0, 
        'subsample': 0.8,
        'early_stopping_rounds': 50
    }
    
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
