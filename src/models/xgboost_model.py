import numpy as np
import xgboost as xgb

class XGBoostQuantileModel:
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        self.quantiles = quantiles
        self.models = {}
        self.params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'verbosity': 0
        }
        
    def fit(self, X_train, y_train):
        for q in self.quantiles:
            params = self.params.copy()
            if q == 0.5:
                params['objective'] = 'reg:squarederror'
            else:
                params['objective'] = 'reg:quantileerror'
                params['quantile_alpha'] = q
            
            try:
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train)
            except:
                dtrain = xgb.DMatrix(X_train, label=y_train)
                if q == 0.5:
                    model = xgb.train(params, dtrain, num_boost_round=params['n_estimators'])
                else:
                    model = xgb.train(params, dtrain, num_boost_round=params['n_estimators'])
                self.models[q] = model
                continue
                
            self.models[q] = model
        return self
    
    def predict(self, X):
        predictions = {}
        for q, model in self.models.items():
            if isinstance(model, xgb.XGBRegressor):
                predictions[q] = model.predict(X)
            else:
                dtest = xgb.DMatrix(X)
                predictions[q] = model.predict(dtest)
        return predictions
    
    def predict_interval(self, X):
        preds = self.predict(X)
        return np.column_stack([preds[0.1], preds[0.5], preds[0.9]])
    
    def save(self, path):
        for q, model in self.models.items():
            if isinstance(model, xgb.XGBRegressor):
                model.save_model(f"{path}_q{int(q*100)}.json")
            else:
                model.save_model(f"{path}_q{int(q*100)}.json")
    
    def load(self, path, quantiles):
        self.quantiles = quantiles
        for q in quantiles:
            model = xgb.XGBRegressor()
            model.load_model(f"{path}_q{int(q*100)}.json")
            self.models[q] = model