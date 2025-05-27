# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from scipy.stats import boxcox, boxcox_normmax, boxcox_llf, chi2
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.stats.diagnostic import acorr_ljungbox, lilliefors
from sklearn.metrics import mean_squared_error
import warnings
import gc
import time
warnings.filterwarnings('ignore')

# Functions
def read_data(pintu_air):
    # Read full dataset
    df = pd.read_csv("https://raw.githubusercontent.com/elvano22/Pintu-Air/refs/heads/main/03%20Result%20Data%20Cleaning%20Part%203.csv")
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df = df.set_index('Tanggal')
    print(f"{'='*60}\nFull Dataset Information:\n{'='*60}")
    print(df.info())
    
    # Read selected pintu air data
    data = df[[f'{pintu_air} (air)', f'{pintu_air} (cuaca)']].copy()
    print(f"\n{'='*60}\n{pintu_air} Dataset:\n{'='*60}")
    print(data.head())

    # Seperate x and y data
    data_y = data[f'{pintu_air} (air)']
    data_x = data[f'{pintu_air} (cuaca)']

    # Seperate train and test data
    split_index = int(len(data) * 0.95)

    data_y_train = data_y[:split_index]
    data_y_test = data_y[split_index:]

    data_x_train = pd.get_dummies(data_x[:split_index], drop_first=False)
    data_x_test = pd.get_dummies(data_x[split_index:], drop_first=False)

    drop_column = 'Terang'
    data_x_train = data_x_train.drop(columns=[drop_column], errors='ignore')
    data_x_test = data_x_test.drop(columns=[drop_column], errors='ignore')
    data_x_test = data_x_test.reindex(columns=data_x_train.columns, fill_value=0)
    print(f"\n{'='*60}\n{pintu_air} Data Train and Test Distribution:\n{'='*60}")
    print(f'\nJumlah data: {len(data_y)}')
    print(f'Jumlah data train: {len(data_y_train)}')
    print(f'Jumlah data test: {len(data_y_test)}')

    return data_x_train, data_x_test, data_y_train, data_y_test

def plot_decompose(decomposition):
    # Create the decomposition plot
    fig, axes = plt.subplots(4, 1, figsize=(500, 25))
    fig.suptitle('Seasonal Decomposition of Pasar Ikan Water Level',
                fontsize=16, fontweight='bold')

    # Original time series
    axes[0].plot(decomposition.observed.index, decomposition.observed.values,
                color='blue', linewidth=0.8)
    axes[0].set_title('Original Time Series')
    axes[0].set_ylabel('Water Level')
    axes[0].grid(True, alpha=0.3)

    # Trend component
    axes[1].plot(decomposition.trend.index, decomposition.trend.values,
                color='red', linewidth=1.2)
    axes[1].set_title('Trend Component')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)

    # Seasonal component
    axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values,
                color='green', linewidth=0.8)
    axes[2].set_title('Seasonal Component (24-hour cycle)')
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)

    # Residual component
    axes[3].plot(decomposition.resid.index, decomposition.resid.values,
                color='purple', linewidth=0.8)
    axes[3].set_title('Residual Component')
    axes[3].set_ylabel('Residuals')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Plot the 4 days seasonal component
    fig, ax = plt.subplots()
    ax.plot(decomposition.seasonal.head(24*4))
    ax.set_title('Seasonal Component (4 days)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    def plot_acf_pacf(data, period, lag):
    fig, axes = plt.subplots(1, 2, figsize=(14, 12))

    # ACF Plot
    for lag in range(period, lag+1, period):
        axes[0].axvline(x=lag, color='red', linestyle='--', linewidth=1, zorder=0)
    plot_acf(data, ax=axes[0], lags=lag, marker='o', markersize=3)
    axes[0].set_title('Autocorrelation Function (ACF)')

    # PACF Plot
    for lag in range(period, lag+1, period):
        axes[1].axvline(x=lag, color='red', linestyle='--', linewidth=1, zorder=0)
    plot_pacf(data, ax=axes[1], lags=lag, marker='o', markersize=3)
    axes[1].set_title('Partial Autocorrelation Function (PACF)')

    plt.tight_layout()
    plt.show()


def boxcox_transformation(data):
    # Hitung lambda optimal
    lambda_opt = boxcox_normmax(data, method='mle')

    # Log-likelihood pada lambda optimal
    llf_opt = boxcox_llf(lambda_opt, data)

    # Log-likelihood jika lambda = 1 (tidak ditransformasi)
    llf_null = boxcox_llf(1.0, data)

    # Hitung LRT
    lrt_stat = 2 * (llf_opt - llf_null)

    # Hitung p-value (df = 1)
    p_value = chi2.sf(lrt_stat, df=1)

    print(f"Lambda optimal: {lambda_opt:.4f}")
    print(f"LRT statistic: {lrt_stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    return lambda_opt

def inverse_transform(transformed_data, lambda_opt):
    if lambda_opt == 0:
        return np.exp(transformed_data)  # Log case: e^(transformed_data)
    elif lambda_opt == 1:
        return transformed_data  # No transformation case
    else:
        return np.power(transformed_data, 1/lambda_opt)
    
def modelling(model_configs, data_y_train_trans, data_y_train, data_x_train, data_y_test, data_x_test, lambda_opt=1.0):
    results_list = []
    
    # Loop through each model configuration
    for config in model_configs:
        try:
            print(f"\n{'='*60}")
            print(f"Testing {config['name']}")
            print(f"{'='*60}")

            # Fit the model based on configuration
            if config['model_type'] in ['ARIMA', 'ARIMAX']:
                if config['exog']:
                    model = SARIMAX(data_y_train_trans, exog=data_x_train, order=config['order'])
                else:
                    model = ARIMA(data_y_train_trans, order=config['order'])
            else:  # SARIMA or SARIMAX
                if config['exog']:
                    model = SARIMAX(data_y_train_trans, exog=data_x_train,
                                order=config['order'], seasonal_order=config['seasonal_order'])
                else:
                    model = SARIMAX(data_y_train_trans, order=config['order'],
                                seasonal_order=config['seasonal_order'])

            # Fit the model
            print("Fitting model...")
            if config['model_type'] in ['ARIMA']:
                results = model.fit()  # Simple fit for ARIMA
            else:
                results = model.fit(disp=False)
            print("Model fitted successfully!")

            # Print model summary
            print(results.summary())

            # Get residuals
            residuals = results.resid

            # Ljung-Box Test
            ljung_box_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
            ljung_box_pvalue = ljung_box_result['lb_pvalue'].iloc[0]
            print(f"\nLjung-Box Test:\n{ljung_box_result}")

            # Lilliefors Test
            stat, lillie_pvalue = lilliefors(residuals)
            print(f"Lilliefors Test: statistic={stat:.4f}, p-value={lillie_pvalue:.4f}")

            # Check if all variables are significant (p-value < 0.05)
            pvalues = results.pvalues
            all_significant = all(pvalues < 0.05)

            # Calculate RMSE for training
            print("Calculating training RMSE...")
            fitted_values = results.fittedvalues
            fitted_values_transformed = inverse_transform(fitted_values, lambda_opt)
            rmse_train = np.sqrt(mean_squared_error(data_y_train, fitted_values_transformed))
            print(f"Training RMSE: {rmse_train:.4f}")

            # Calculate RMSE for testing
            print("Calculating testing RMSE...")
            try:
                if config['exog']:
                    forecast = results.forecast(steps=len(data_y_test), exog=data_x_test)
                else:
                    forecast = results.forecast(steps=len(data_y_test))
                forecast_transformed = inverse_transform(forecast, lambda_opt)
                rmse_test = np.sqrt(mean_squared_error(data_y_test, forecast_transformed))
                print(f"Testing RMSE: {rmse_test:.4f}")
            except Exception as forecast_error:
                print(f"Forecasting failed: {forecast_error}")
                rmse_test = np.nan

            # Store results
            results_dict = {
                'Nama Dataset': data_y_train.name,
                'Model': config['name'],
                'Semua Variabel Signifikan': 'Ya' if all_significant else 'Tidak',
                'P-value Ljung Box': f"{ljung_box_pvalue:.6f}",
                'White Noise?': 'Ya' if ljung_box_pvalue > 0.05 else 'Tidak',
                'P-value Lillie Test': f"{lillie_pvalue:.6f}",
                'Distribusi Normal?': 'Ya' if lillie_pvalue > 0.05 else 'Tidak',
                'RMSE Training': f"{rmse_train:.4f}",
                'RMSE Testing': f"{rmse_test:.4f}" if not np.isnan(rmse_test) else 'Error'
            }

            results_list.append(results_dict)
            print(f"Model {config['name']} completed successfully!")

            # MEMORY CLEANUP - Delete large objects after each successful model
            del model, results, fitted_values, fitted_values_transformed
            if 'forecast' in locals():
                del forecast
            if 'forecast_transformed' in locals():
                del forecast_transformed
            if 'residuals' in locals():
                del residuals
            if 'ljung_box_result' in locals():
                del ljung_box_result
            
            # Force garbage collection to free memory immediately
            collected = gc.collect()
            print(f"Memory cleanup: {collected} objects collected")

        except Exception as e:
            print(f"Error fitting {config['name']}: {str(e)}")
            
            # Still add to results with error indicators
            results_dict = {
                'Nama Dataset': getattr(data_y_train_trans, 'name', 'Unknown'),
                'Model': config['name'],
                'Semua Variabel Signifikan': 'Error',
                'P-value Ljung Box': 'Error',
                'White Noise?': 'Error',
                'P-value Lillie Test': 'Error',
                'Distribusi Normal?': 'Error',
                'RMSE Training': 'Error',
                'RMSE Testing': 'Error'
            }
            results_list.append(results_dict)
            
            # MEMORY CLEANUP - Even on error, clean up any created objects
            if 'model' in locals():
                del model
            if 'results' in locals():
                del results
            if 'fitted_values' in locals():
                del fitted_values
            if 'fitted_values_transformed' in locals():
                del fitted_values_transformed
            if 'forecast' in locals():
                del forecast
            if 'forecast_transformed' in locals():
                del forecast_transformed
            if 'residuals' in locals():
                del residuals
            if 'ljung_box_result' in locals():
                del ljung_box_result
            
            # Force garbage collection after error
            collected = gc.collect()
            print(f"Error cleanup: {collected} objects collected")

    # FINAL CLEANUP - Clean up any remaining objects
    print(f"\n{'='*60}")
    print("Final memory cleanup...")
    final_collected = gc.collect()
    print(f"Final cleanup: {final_collected} objects collected")
    print("All models completed!")
    print(f"{'='*60}")

    return pd.DataFrame(results_list)
