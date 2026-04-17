import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.feature_columns = None
        self.target_column = 'days_to_payment'
        self.date_column = 'invoice_date'
        
    def load(self):
        self.data = pd.read_csv(self.filepath)
        self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
        self.data = self.data.sort_values(self.date_column)
        self.feature_columns = [col for col in self.data.columns 
                                if col not in ['days_to_payment', 'invoice_id', 
                                               'customer_id', 'invoice_date', 
                                               'due_date', 'payment_date']]
        return self.data
    
    def create_scarcity_windows(self, months_list):
        windows = {}
        max_date = self.data[self.date_column].max()
        
        for months in months_list:
            cutoff_date = max_date - pd.DateOffset(months=months)
            window_data = self.data[self.data[self.date_column] >= cutoff_date].copy()
            windows[months] = window_data
            
        return windows
    
    def prepare_features_target(self, data):
        X = data[self.feature_columns].values
        y = data[self.target_column].values
        return X, y
    
    def get_train_test_split(self, data, test_size=0.2):
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        X_train, y_train = self.prepare_features_target(train_data)
        X_test, y_test = self.prepare_features_target(test_data)
        
        return X_train, X_test, y_train, y_test, test_data