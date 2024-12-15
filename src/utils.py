import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_sales_trend(data, material_id=None):
    """Plot sales trend over time"""
    plt.figure(figsize=(15, 6))
    if material_id:
        material_data = data[data['material_id'] == material_id]
        sns.lineplot(data=material_data, x='shipping_date', y='std_shipping_quantity')
        plt.title(f'Sales Trend for Material {material_id}')
    else:
        sns.lineplot(data=data, x='shipping_date', y='std_shipping_quantity')
        plt.title('Overall Sales Trend')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def plot_forecast(forecast, actual=None, material_id=None):
    """Plot forecast with confidence intervals"""
    plt.figure(figsize=(15, 6))
    
    # Plot forecast
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    plt.fill_between(forecast['ds'], 
                    forecast['yhat_lower'], 
                    forecast['yhat_upper'], 
                    alpha=0.3,
                    label='Confidence Interval')
    
    # Plot actual values if provided
    if actual is not None:
        plt.plot(actual['ds'], actual['y'], 'r.', label='Actual')
    
    plt.title(f'Sales Forecast{" for Material " + str(material_id) if material_id else ""}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def save_forecasts(forecasts, output_dir):
    """Save forecasts to CSV files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for material_id, forecast in forecasts.items():
        forecast.to_csv(output_path / f'forecast_material_{material_id}.csv', index=False)
        
def calculate_forecast_accuracy(actual, predicted):
    """Calculate forecast accuracy metrics"""
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    return {
        'mape': mape,
        'mae': mae,
        'rmse': rmse
    }
