import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FeatureEngineer:
    def __init__(self):
        pass
        
    def create_time_features(self, df, date_column):
        """Create time-based features from date column"""
        df = df.copy()
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['quarter'] = df[date_column].dt.quarter
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['day_of_month'] = df[date_column].dt.day
        df['week_of_year'] = df[date_column].dt.isocalendar().week
        return df
    
    def aggregate_sales_data(self, sales_df, groupby_cols=['material_id', 'shipping_date']):
        """Aggregate sales data by specified columns"""
        return sales_df.groupby(groupby_cols)['std_shipping_quantity'].sum().reset_index()
    
    def merge_inventory_features(self, sales_df, inventory_df):
        """Merge inventory levels with sales data"""
        inventory_agg = inventory_df.groupby(['material_id', 'date'])['inventory_quantity'].mean().reset_index()
        return pd.merge_asof(
            sales_df.sort_values('shipping_date'),
            inventory_agg.sort_values('date'),
            left_on='shipping_date',
            right_on='date',
            by='material_id'
        )
    
    def calculate_moving_averages(self, df, target_col, windows=[7, 30, 90]):
        """Calculate moving averages for different time windows"""
        df = df.sort_values('shipping_date')
        for window in windows:
            df[f'ma_{window}d'] = df.groupby('material_id')[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        return df
    
    def add_lag_features(self, df, target_col, lags=[1, 7, 30]):
        """Add lagged features for the target column"""
        df = df.sort_values('shipping_date')
        for lag in lags:
            df[f'lag_{lag}d'] = df.groupby('material_id')[target_col].shift(lag)
        return df
