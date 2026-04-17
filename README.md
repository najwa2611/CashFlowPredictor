# Cash Flow Predictor

Ce projet est un outil de prévision du cash flow pour des factures clients. Il combine :
- génération de données synthétiques de factures (`generate_data.py`)
- entraînement et backtesting de modèles de prédiction de délais de paiement (`main.py`)
- prédictions probabilistes P10 / P50 / P90 avec un modèle XGBoost quantile
- une application web Flask pour charger un CSV de factures et obtenir un rapport de risque

## 🚀 Fonctions principales

- `generate_data.py` : génère un jeu de données synthétique `data/invoices.csv`
- `main.py` : charge les données, exécute un backtesting walk-forward, génère un rapport CFO et sauvegarde le meilleur modèle
- `app.py` : lance une interface web de type dashboard pour prédire les délais et niveaux de risque par facture

## 🧠 Modèles

- `src/models/xgboost_model.py` : modèle quantile XGBoost pour les prédictions P10/P50/P90
- `src/models/random_forest_model.py` : modèle forêt aléatoire pour les intervalles de confiance

## 📦 Installation

1. Créez un environnement virtuel :
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## 📝 Utilisation

### 1. Générer des données synthétiques

Si vous n'avez pas de fichier de factures, lancez :
```bash
python generate_data.py
```
Cela crée `data/invoices.csv` avec des colonnes de factures et des variables de délai de paiement.

### 2. Entraîner et backtester les modèles

Pour exécuter le pipeline principal et générer les livrables :
```bash
python main.py
```

Sorties attendues :
- `outputs/models/` : modèles sauvegardés
- `outputs/figures/` : graphiques d'analyse
- `outputs/results/` : rapports CRPS et CFO

### 3. Lancer l'application web

Démarrez le serveur Flask :
```bash
python app.py
```

Ouvrez ensuite votre navigateur sur :

`http://127.0.0.1:5000/`

Vous pouvez charger un fichier CSV de factures et obtenir :
- prévisions de délai de paiement P10/P50/P90
- score de risque
- résumé global
- téléchargement du rapport de prédiction

## 📁 Structure du projet

- `app.py` : application Flask et interface web
- `main.py` : exécution du backtesting, évaluation et génération de rapports
- `generate_data.py` : génération de données factices
- `requirements.txt` : dépendances Python
- `data/` : données sources
- `outputs/` : résultats générés
  - `figures/` : graphiques
  - `models/` : modèles sauvegardés
  - `results/` : tables CRPS, rapports, etc.
- `src/`
  - `evaluation/` : modules d'évaluation, backtesting et explicabilité
  - `models/` : implémentation des modèles prédictifs
  - `utils/` : chargement et préparation des données

## 📄 Format du fichier CSV attendu par l'application

Le fichier doit contenir au minimum les colonnes suivantes :

- `amount`
- `payment_terms_days`
- `ar_aging_days`
- `customer_avg_historical_delay`
- `customer_payment_probability`
- `month`
- `quarter`
- `day_of_week`
- `is_month_end`
- `customer_invoice_count`
- `size_large`
- `size_medium`
- `size_small`

## 🔧 Notes

- Le modèle XGBoost quantile charge actuellement les fichiers `outputs/models/xgboost_best_q10.json`, `xgboost_best_q50.json` et `xgboost_best_q90.json`.
- L'application web attend un CSV propre avec les caractéristiques déjà préparées.