import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_loader import DataLoader
from src.models.xgboost_model import XGBoostQuantileModel
from src.models.random_forest_model import RandomForestIntervalModel
from src.evaluation.crps import CRPSEvaluator
from src.evaluation.backtesting import WalkForwardBacktester
from src.evaluation.shap_explainer import SHAPExplainer

def main():
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)
    
    loader = DataLoader('data/invoices.csv')
    data = loader.load()
    
    print(f"Données chargées: {len(data)} factures")
    print(f"Features disponibles: {loader.feature_columns}")
    
    backtester = WalkForwardBacktester(loader, model_types=['xgboost', 'rf'])
    scarcity_months = [3, 6, 9, 12]
    
    print("\nExécution du Walk-Forward Backtesting...")
    results = backtester.run_backtest(scarcity_months)
    
    crps_table = backtester.generate_crps_table()
    
    for col in crps_table.columns:
        if col != 'Scarcity_Months':
            crps_table[col] = crps_table[col].apply(lambda x: x[0] if isinstance(x, (list, tuple, np.ndarray)) else x)
    
    crps_table.to_csv('outputs/results/crps_table.csv', index=False)
    print("\nTable CRPS:")
    print(crps_table)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    crps_data = {}
    for months in scarcity_months:
        for model_type in ['xgboost', 'rf']:
            if model_type not in crps_data:
                crps_data[model_type] = []
            
            crps_value = results[months][model_type]['crps']
            if isinstance(crps_value, (list, tuple, np.ndarray)):
                crps_value = crps_value[0]
            crps_data[model_type].append(crps_value)
    
    for model_type in ['xgboost', 'rf']:
        ax.plot(scarcity_months, crps_data[model_type], marker='o', label=model_type.upper(), linewidth=2)
    
    ax.set_xlabel('Mois d\'historique disponibles')
    ax.set_ylabel('CRPS Score')
    ax.set_title('Courbe de Dégradation - Impact de la Scarcité des Données')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/figures/degradation_curve.png', dpi=300)
    plt.show()
    
    crps_values_xgb = crps_data['xgboost']
    threshold_idx = np.argmin(np.abs(np.array(crps_values_xgb)))
    viable_threshold = scarcity_months[threshold_idx]
    print(f"\nSeuil minimum viable: {viable_threshold} mois d'historique")
    
    best_model_data = results[viable_threshold]['xgboost']
    best_model = best_model_data['model']
    
    X_train, X_test, y_train, y_test, test_data = loader.get_train_test_split(data)
    
    explainer = SHAPExplainer(best_model, loader.feature_columns)
    shap_values = explainer.explain_xgboost(X_train[:100], X_test[:20])
    
    explainer.generate_summary_plot('outputs/figures/shap_summary.png')
    
    sample_invoices = test_data.iloc[:5]
    sample_features = X_test[:5]
    
    customer_names = sample_invoices['customer_id'].values if 'customer_id' in sample_invoices.columns else None
    cfo_report = explainer.cfo_readable_report(sample_features, customer_names=customer_names)
    
    print("\nRapport CFO - Top 5 factures:")
    for line in cfo_report:
        print(f"  • {line}")
    
    with open('outputs/results/cfo_report.txt', 'w', encoding='utf-8') as f:
        f.write("RAPPORT CFO - ANALYSE DES RISQUES DE PAIEMENT\n")
        f.write("="*50 + "\n\n")
        for line in cfo_report:
            f.write(f"• {line}\n")
    
    best_model.save('outputs/models/xgboost_best')
    
    print("\nProjet terminé. Tous les livrables sont dans /outputs/")

if __name__ == "__main__":
    main()