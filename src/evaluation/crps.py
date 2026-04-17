import numpy as np
from properscoring import crps_ensemble

class CRPSEvaluator:
    def __init__(self):
        self.scores = {}
        
    def compute_crps(self, y_true, y_pred_distribution):
        return crps_ensemble(y_true, y_pred_distribution)
    
    def evaluate_model(self, y_true, predictions_dict):
        pred_matrix = np.column_stack([
            predictions_dict[0.1],
            predictions_dict[0.5],
            predictions_dict[0.9]
        ])
        
        crps_value = self.compute_crps(y_true, pred_matrix)
        return crps_value
    
    def compute_degradation_curve(self, results_by_window):
        windows = sorted(results_by_window.keys())
        crps_values = [results_by_window[w]['crps'] for w in windows]
        
        return {
            'windows': windows,
            'crps_values': crps_values
        }