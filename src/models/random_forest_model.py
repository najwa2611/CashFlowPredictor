import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForestIntervalModel:
    def __init__(self, n_estimators=200):
        self.n_estimators = n_estimators
        self.model = None
        self.residuals_std = None
        
    def fit(self, X_train, y_train):
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_train)
        residuals = y_train - y_pred
        self.residuals_std = np.std(residuals)
        
        return self
    
    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred
    
    def predict_interval(self, X, confidence=0.8):
        y_pred = self.predict(X)
        z_score = 1.28
        margin = z_score * self.residuals_std
        
        p10 = y_pred - margin
        p50 = y_pred
        p90 = y_pred + margin
        
        return np.column_stack([p10, p50, p90])
    
    def save(self, path):
        import joblib
        joblib.dump({'model': self.model, 'residuals_std': self.residuals_std}, path)
    
    def load(self, path):
        import joblib
        data = joblib.load(path)
        self.model = data['model']
        self.residuals_std = data['residuals_std']