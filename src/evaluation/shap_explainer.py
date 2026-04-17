import shap
import numpy as np
import matplotlib.pyplot as plt

class SHAPExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def explain_xgboost(self, X_background, X_explain):
        X_background = np.array(X_background, dtype=np.float64)
        X_explain = np.array(X_explain, dtype=np.float64)
        
        try:
            self.explainer = shap.TreeExplainer(
                self.model.models[0.5],
                data=X_background[:100],
                feature_perturbation="interventional"
            )
        except:
            try:
                self.explainer = shap.TreeExplainer(
                    self.model.models[0.5],
                    feature_perturbation="tree_path_dependent"
                )
            except:
                self.explainer = shap.KernelExplainer(
                    self.model.models[0.5].predict, 
                    X_background[:50]
                )
                self.shap_values = self.explainer.shap_values(X_explain, nsamples=100)
                return self.shap_values
        
        self.shap_values = self.explainer.shap_values(X_explain)
        
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[0]
        
        return self.shap_values
    
    def explain_random_forest(self, X_background, X_explain):
        X_background = np.array(X_background, dtype=np.float64)
        X_explain = np.array(X_explain, dtype=np.float64)
        
        try:
            self.explainer = shap.TreeExplainer(
                self.model.model,
                data=X_background[:100],
                feature_perturbation="interventional"
            )
        except:
            try:
                self.explainer = shap.TreeExplainer(
                    self.model.model,
                    feature_perturbation="tree_path_dependent"
                )
            except:
                self.explainer = shap.KernelExplainer(
                    self.model.model.predict, 
                    X_background[:50]
                )
                self.shap_values = self.explainer.shap_values(X_explain, nsamples=100)
                return self.shap_values
        
        self.shap_values = self.explainer.shap_values(X_explain)
        
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[0]
        
        return self.shap_values
    
    def generate_summary_plot(self, save_path):
        if self.shap_values is None:
            print("Pas de valeurs SHAP à afficher")
            return
        
        plt.figure(figsize=(10, 8))
        
        try:
            shap.summary_plot(
                self.shap_values, 
                feature_names=self.feature_names,
                show=False
            )
        except:
            shap.summary_plot(
                self.shap_values, 
                show=False
            )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_waterfall_plot(self, index, save_path):
        if self.shap_values is None:
            print("Pas de valeurs SHAP à afficher")
            return
        
        plt.figure(figsize=(10, 6))
        
        try:
            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values[index],
                    base_values=np.mean(self.shap_values),
                    feature_names=self.feature_names
                ),
                show=False
            )
        except:
            pass
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_top_features_per_invoice(self, X_explain, top_k=3):
        if self.shap_values is None:
            return []
        
        explanations = []
        
        for i in range(min(len(X_explain), len(self.shap_values))):
            invoice_shap = self.shap_values[i]
            top_indices = np.argsort(np.abs(invoice_shap))[-top_k:][::-1]
            
            invoice_exp = {
                'invoice_index': i,
                'top_features': []
            }
            
            for idx in top_indices:
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                else:
                    feature_name = f"Feature_{idx}"
                
                shap_value = invoice_shap[idx]
                impact_direction = 'delay' if shap_value > 0 else 'advance'
                
                invoice_exp['top_features'].append({
                    'feature': feature_name,
                    'shap_value': float(shap_value),
                    'impact': impact_direction
                })
            
            explanations.append(invoice_exp)
        
        return explanations
    
    def cfo_readable_report(self, X_explain, customer_names=None):
        explanations = self.get_top_features_per_invoice(X_explain)
        report_lines = []
        
        if not explanations:
            return ["SHAP non disponible - Utiliser XGBoost 1.7.6 pour SHAP complet"]
        
        for i, exp in enumerate(explanations):
            if customer_names is not None and i < len(customer_names):
                customer = customer_names[i]
            else:
                customer = f"Facture_{i}"
            
            line = f"{customer}: "
            
            if exp['top_features']:
                for feat in exp['top_features']:
                    impact_amount = abs(feat['shap_value']) * 1000
                    if feat['impact'] == 'delay':
                        line += f"retard de €{impact_amount:.0f} dû à {feat['feature']}, "
                    else:
                        line += f"avance de €{impact_amount:.0f} grâce à {feat['feature']}, "
                line = line.rstrip(', ')
            else:
                line += "pas de facteurs significatifs"
            
            report_lines.append(line)
        
        return report_lines