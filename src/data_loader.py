import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        
    def load_all_data(self):
        """Load all datasets and return them as a dictionary"""
        datasets = {
            'sales': self.load_sales_data(),
            'delivery': self.load_delivery_data(),
            'material': self.load_material_data(),
            'inventory': self.load_inventory_data()
        }
        return datasets
    
    def load_sales_data(self):
        """Load and preprocess sales data"""
        df = pd.read_csv(self.data_dir / 'sales_data.csv')
        df['shipping_date'] = pd.to_datetime(df['shipping_date'])
        return df
    
    def load_delivery_data(self):
        """Load and preprocess delivery data"""
        df = pd.read_csv(self.data_dir / 'delivery_data.csv')
        df['delivery_date'] = pd.to_datetime(df['delivery_date'])
        return df
    
    def load_material_data(self):
        """Load and preprocess material data"""
        return pd.read_csv(self.data_dir / 'material_data.csv')
    
    def load_inventory_data(self):
        """Load and preprocess inventory data"""
        df = pd.read_csv(self.data_dir / 'inventory_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
