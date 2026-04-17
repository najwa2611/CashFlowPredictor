import numpy as np
import pandas as pd
from ..models import XGBoostQuantileModel, RandomForestIntervalModel, LSTMMCDropoutModel
from .crps import CRPSEvaluator

class WalkForwardBacktester:
    def __init__(self, data_loader, model_types=['xgboost', 'rf', 'lstm']):
        self.data_loader = data_loader
        self.model_types = model_types
        self.crps_evaluator = CRPSEvaluator()
        self.results = {}
        
    def run_backtest(self, scarcity_months):
        all_results = {}
        
        for months in scarcity_months:
            window_data = self.data_loader.create_scarcity_windows([months])[months]
            X, y = self.data_loader.prepare_features_target(window_data)
            
            n_samples = len(X)
            train_size = int(n_samples * 0.7)
            val_size = int(n_samples * 0.15)
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size+val_size]
            y_val = y[train_size:train_size+val_size]
            X_test = X[train_size+val_size:]
            y_test = y[train_size+val_size:]
            
            month_results = {}
            
            if 'xgboost' in self.model_types:
                xgb_model = XGBoostQuantileModel()
                xgb_model.fit(X_train, y_train)
                xgb_preds = xgb_model.predict(X_test)
                xgb_crps = self.crps_evaluator.evaluate_model(y_test, xgb_preds)
                month_results['xgboost'] = {'crps': xgb_crps, 'model': xgb_model}
            
            if 'rf' in self.model_types:
                rf_model = RandomForestIntervalModel()
                rf_model.fit(X_train, y_train)
                rf_interval = rf_model.predict_interval(X_test)
                rf_preds = {0.1: rf_interval[:, 0], 0.5: rf_interval[:, 1], 0.9: rf_interval[:, 2]}
                rf_crps = self.crps_evaluator.evaluate_model(y_test, rf_preds)
                month_results['rf'] = {'crps': rf_crps, 'model': rf_model}
            
            if 'lstm' in self.model_types:
                lstm_model = LSTMMCDropoutModel(input_size=X.shape[1])
                lstm_model.fit(X_train, y_train, epochs=30)
                lstm_preds_all = []
                for i in range(len(X_test)):
                    X_hist = np.vstack([X_train, X_val, X_test[:i+1]])
                    pred = lstm_model.predict_interval(X_hist)
                    lstm_preds_all.append(pred[0])
                lstm_preds_all = np.array(lstm_preds_all)
                lstm_preds = {
                    0.1: lstm_preds_all[:, 0],
                    0.5: lstm_preds_all[:, 1],
                    0.9: lstm_preds_all[:, 2]
                }
                lstm_crps = self.crps_evaluator.evaluate_model(y_test, lstm_preds)
                month_results['lstm'] = {'crps': lstm_crps, 'model': lstm_model}
            
            all_results[months] = month_results
        
        self.results = all_results
        return all_results
    
    def generate_crps_table(self):
        table_data = []
        
        for months, month_results in self.results.items():
            row = [months]
            for model_type in self.model_types:
                if model_type in month_results:
                    row.append(month_results[model_type]['crps'])
                else:
                    row.append(np.nan)
            table_data.append(row)
        
        columns = ['Scarcity_Months'] + self.model_types
        crps_table = pd.DataFrame(table_data, columns=columns)
        
        return crps_table