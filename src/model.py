import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
import logging
from pathlib import Path

from data_loader import DataLoader
from feature_eng import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesForecaster:
    def __init__(self, data_dir='data'):
        self.data_loader = DataLoader(data_dir)
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        
    def prepare_data(self):
        """Load and prepare data for modeling"""
        logger.info("Loading data...")
        data = self.data_loader.load_all_data()
        
        # Aggregate sales data by date and material
        sales_agg = self.feature_engineer.aggregate_sales_data(data['sales'])
        
        # Add time-based features
        sales_features = self.feature_engineer.create_time_features(sales_agg, 'shipping_date')
        
        # Add inventory features
        sales_features = self.feature_engineer.merge_inventory_features(
            sales_features, data['inventory']
        )
        
        # Add moving averages and lag features
        sales_features = self.feature_engineer.calculate_moving_averages(
            sales_features, 'std_shipping_quantity'
        )
        sales_features = self.feature_engineer.add_lag_features(
            sales_features, 'std_shipping_quantity'
        )
        
        return sales_features.dropna()
    
    def train_prophet_model(self, data, material_id):
        """Train Prophet model for a specific material"""
        # Prepare data for Prophet
        prophet_data = data[data['material_id'] == material_id].copy()
        prophet_data = prophet_data.rename(columns={
            'shipping_date': 'ds',
            'std_shipping_quantity': 'y'
        })
        
        # Initialize and train Prophet model
        model = Prophet(
            yearly_seasonality = True,
            weekly_seasonality = True,
            daily_seasonality = False,
            seasonality_mode='multiplicative'
        )
        model.fit(prophet_data)
        return model
    
    def train_models(self):
        """Train forecasting models for each material"""
        data = self.prepare_data()
        
        logger.info("Training models...")
        for material_id in data['material_id'].unique():
            logger.info(f"Training model for material_id: {material_id}")
            self.models[material_id] = self.train_prophet_model(data, material_id)
    
    def make_predictions(self, periods=30):
        """Generate forecasts for all materials"""
        forecasts = {}
        
        for material_id, model in self.models.items():
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            forecasts[material_id] = forecast
        
        return forecasts
    
    def evaluate_models(self):
        """Evaluate model performance"""
        metrics = {}
        data = self.prepare_data()
        
        for material_id, model in self.models.items():
            material_data = data[data['material_id'] == material_id].copy()
            material_data = material_data.rename(columns={'shipping_date': 'ds', 'std_shipping_quantity': 'y'})
            
            # Make predictions on historical data
            predictions = model.predict(material_data[['ds']])
            
            # Calculate metrics
            metrics[material_id] = {
                'mae': mean_absolute_error(material_data['y'], predictions['yhat']),
                'rmse': np.sqrt(mean_squared_error(material_data['y'], predictions['yhat'])),
                'r2': r2_score(material_data['y'], predictions['yhat'])
            }
        
        return metrics

def main():
    forecaster = SalesForecaster()
    forecaster.train_models()
    
    # Generate forecasts
    forecasts = forecaster.make_predictions()
    
    # Evaluate models
    metrics = forecaster.evaluate_models()
    
    logger.info("Model evaluation metrics:")
    for material_id, metric in metrics.items():
        logger.info(f"Material {material_id}:")
        logger.info(f"  MAE: {metric['mae']:.2f}")
        logger.info(f"  RMSE: {metric['rmse']:.2f}")
        logger.info(f"  R2: {metric['r2']:.2f}")

if __name__ == "__main__":
    main()
