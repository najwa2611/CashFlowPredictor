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
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cash Flow Predictor - CFO Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; }
            .upload-area { border: 2px dashed #3498db; border-radius: 5px; padding: 40px; text-align: center; margin: 20px 0; }
            .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            .btn:hover { background: #2980b9; }
            #result { margin-top: 20px; display: none; }
            .loading { color: #7f8c8d; }
            .success { color: #27ae60; }
            .error { color: #e74c3c; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background: #34495e; color: white; }
            .risk-high { background: #fee; }
            .risk-medium { background: #fff3cd; }
            .risk-low { background: #d4edda; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Cash Flow Predictor</h1>
            <p>Prévisions probabilistes P10/P50/P90 pour vos factures</p>
            
            <div class="upload-area">
                <h3>📁 Charger votre fichier CSV</h3>
                <p>Format attendu: factures avec colonnes standard</p>
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="file" name="file" accept=".csv" required style="margin: 20px 0;">
                    <br>
                    <button type="submit" class="btn">🚀 Analyser les prévisions</button>
                </form>
            </div>
            
            <div id="result">
                <div id="message"></div>
                <div id="summary"></div>
                <div id="table"></div>
                <div id="download"></div>
            </div>
        </div>
        
        <script>
            document.getElementById('uploadForm').onsubmit = async (e) => {
                e.preventDefault();
                
                const resultDiv = document.getElementById('result');
                const messageDiv = document.getElementById('message');
                resultDiv.style.display = 'block';
                messageDiv.innerHTML = '<p class="loading">⏳ Analyse en cours...</p>';
                
                const formData = new FormData();
                formData.append('file', document.getElementById('file').files[0]);
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        messageDiv.innerHTML = `<p class="error">❌ ${data.error}</p>`;
                    } else {
                        messageDiv.innerHTML = `<p class="success">✅ ${data.message}</p>`;
                        
                        document.getElementById('summary').innerHTML = `
                            <h3>📈 Résumé</h3>
                            <p>Total factures: ${data.summary.total_invoices}</p>
                            <p>Montant total: €${data.summary.total_amount.toLocaleString()}</p>
                            <p>Cash flow à risque (retard probable): €${data.summary.at_risk_amount.toLocaleString()}</p>
                            <p>Délai moyen prévu (P50): ${data.summary.avg_delay.toFixed(1)} jours</p>
                        `;
                        
                        let tableHtml = '<h3>📋 Top 10 factures à risque</h3><table><tr><th>Facture</th><th>Client</th><th>Montant</th><th>P10</th><th>P50</th><th>P90</th><th>Risque</th></tr>';
                        
                        data.predictions.forEach(p => {
                            const riskClass = p.risk_level === 'HIGH' ? 'risk-high' : 
                                            p.risk_level === 'MEDIUM' ? 'risk-medium' : 'risk-low';
                            tableHtml += `<tr class="${riskClass}">
                                <td>${p.invoice_id}</td>
                                <td>${p.customer}</td>
                                <td>€${p.amount.toLocaleString()}</td>
                                <td>${p.P10.toFixed(1)}j</td>
                                <td>${p.P50.toFixed(1)}j</td>
                                <td>${p.P90.toFixed(1)}j</td>
                                <td>${p.risk_level}</td>
                            </tr>`;
                        });
                        
                        tableHtml += '</table>';
                        document.getElementById('table').innerHTML = tableHtml;
                        
                        document.getElementById('download').innerHTML = `
                            <br>
                            <a href="/download/${data.filename}" class="btn">📥 Télécharger le rapport complet (CSV)</a>
                        `;
                    }
                } catch (error) {
                    messageDiv.innerHTML = `<p class="error">❌ Erreur: ${error.message}</p>`;
                }
            };
        </script>
    </body>
    </html>
    '''

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