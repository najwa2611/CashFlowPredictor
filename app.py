import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from src.models.xgboost_model import XGBoostQuantileModel
from src.utils.data_loader import DataLoader
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs/web', exist_ok=True)

model = XGBoostQuantileModel()
model.load('outputs/models/xgboost_best', quantiles=[0.1, 0.5, 0.9])

feature_columns = ['amount', 'payment_terms_days', 'ar_aging_days', 
                   'customer_avg_historical_delay', 'customer_payment_probability',
                   'month', 'quarter', 'day_of_week', 'is_month_end',
                   'customer_invoice_count', 'size_large', 'size_medium', 'size_small']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier uploadé'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide'})
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Format CSV requis'})
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        df = pd.read_csv(filepath)
        
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            return jsonify({'error': f'Colonnes manquantes: {missing_cols}'})
        
        X = df[feature_columns].values
        predictions = model.predict(X)
        
        results_df = df.copy()
        results_df['P10_days'] = predictions[0.1]
        results_df['P50_days'] = predictions[0.5]
        results_df['P90_days'] = predictions[0.9]
        results_df['expected_delay'] = results_df['P50_days']
        results_df['risk_score'] = results_df['P90_days'] - results_df['P10_days']
        
        def risk_level(row):
            if row['P50_days'] > 30 or row['P90_days'] > 60:
                return 'HIGH'
            elif row['P50_days'] > 15:
                return 'MEDIUM'
            return 'LOW'
        
        results_df['risk_level'] = results_df.apply(risk_level, axis=1)
        
        output_filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path = os.path.join('outputs/web', output_filename)
        results_df.to_csv(output_path, index=False)
        
        risk_df = results_df.nlargest(10, 'risk_score')
        
        summary = {
            'total_invoices': len(results_df),
            'total_amount': float(results_df['amount'].sum()) if 'amount' in results_df else 0,
            'at_risk_amount': float(results_df[results_df['risk_level'] == 'HIGH']['amount'].sum()) if 'amount' in results_df else 0,
            'avg_delay': float(results_df['P50_days'].mean())
        }
        
        predictions_list = []
        for _, row in risk_df.iterrows():
            predictions_list.append({
                'invoice_id': str(row.get('invoice_id', 'N/A')),
                'customer': str(row.get('customer_id', 'N/A')),
                'amount': float(row.get('amount', 0)),
                'P10': float(row['P10_days']),
                'P50': float(row['P50_days']),
                'P90': float(row['P90_days']),
                'risk_level': row['risk_level']
            })
        
        return jsonify({
            'message': f'Analyse terminée. {len(results_df)} factures traitées.',
            'summary': summary,
            'predictions': predictions_list,
            'filename': output_filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download/<filename>')
def download(filename):
    return send_file(
        os.path.join('outputs/web', filename),
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    print("\n🚀 Serveur démarré: http://localhost:5000")
    print("📁 Upload votre CSV pour obtenir les prédictions P10/P50/P90\n")
    app.run(debug=True, host='0.0.0.0', port=5000)