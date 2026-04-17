import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

os.makedirs('data', exist_ok=True)

np.random.seed(42)
n_invoices = 50000

customer_ids = [f'CUST_{i:04d}' for i in range(1, 501)]
customers = np.random.choice(customer_ids, n_invoices)

base_date = datetime(2024, 1, 1)
invoice_dates = [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_invoices)]

amounts = np.random.lognormal(mean=7.5, sigma=1.2, size=n_invoices)
amounts = np.round(amounts, 2)

payment_terms = np.random.choice([15, 30, 45, 60], n_invoices, p=[0.2, 0.5, 0.2, 0.1])
due_dates = [inv_date + timedelta(days=int(term)) for inv_date, term in zip(invoice_dates, payment_terms)]

customer_behavior = {}
for cust in customer_ids:
    customer_behavior[cust] = {
        'avg_delay': np.random.uniform(-5, 25),
        'payment_prob': np.random.uniform(0.7, 1.0)
    }

days_to_payment = []
payment_dates = []
ar_aging = []
customer_avg_delay = []
customer_payment_score = []
invoice_size_category = []

for i in range(n_invoices):
    cust = customers[i]
    due = due_dates[i]
    term = payment_terms[i]
    amt = amounts[i]
    
    behavior = customer_behavior[cust]
    base_delay = behavior['avg_delay']
    
    if amt > 10000:
        delay_modifier = np.random.uniform(2, 8)
    elif amt > 5000:
        delay_modifier = np.random.uniform(0, 4)
    else:
        delay_modifier = np.random.uniform(-3, 2)
    
    if term == 15:
        delay_modifier += np.random.uniform(-2, 1)
    elif term == 60:
        delay_modifier += np.random.uniform(1, 5)
    
    actual_delay = base_delay + delay_modifier + np.random.normal(0, 3)
    actual_delay = max(-10, min(60, actual_delay))
    
    days_to_payment.append(actual_delay)
    payment_date = due + timedelta(days=actual_delay)
    payment_dates.append(payment_date)
    
    aging = (datetime.now() - due).days
    ar_aging.append(max(0, aging) if aging > 0 else 0)
    
    customer_avg_delay.append(base_delay)
    customer_payment_score.append(behavior['payment_prob'])
    
    if amt < 1000:
        invoice_size_category.append('small')
    elif amt < 5000:
        invoice_size_category.append('medium')
    else:
        invoice_size_category.append('large')

df = pd.DataFrame({
    'invoice_id': [f'INV_{i:06d}' for i in range(1, n_invoices + 1)],
    'customer_id': customers,
    'invoice_date': invoice_dates,
    'due_date': due_dates,
    'payment_date': payment_dates,
    'amount': amounts,
    'payment_terms_days': payment_terms,
    'days_to_payment': days_to_payment,
    'ar_aging_days': ar_aging,
    'customer_avg_historical_delay': customer_avg_delay,
    'customer_payment_probability': customer_payment_score,
    'invoice_size_category': invoice_size_category,
    'month': [d.month for d in invoice_dates],
    'quarter': [((d.month - 1) // 3) + 1 for d in invoice_dates],
    'day_of_week': [d.weekday() for d in invoice_dates],
    'is_month_end': [1 if d.day >= 28 else 0 for d in invoice_dates],
    'customer_invoice_count': np.random.randint(1, 50, n_invoices)
})

df['invoice_date'] = df['invoice_date'].dt.strftime('%Y-%m-%d')
df['due_date'] = df['due_date'].dt.strftime('%Y-%m-%d')
df['payment_date'] = df['payment_date'].dt.strftime('%Y-%m-%d')

df = pd.get_dummies(df, columns=['invoice_size_category'], prefix='size')

df.to_csv('data/invoices.csv', index=False)

print(f"Fichier créé: data/invoices.csv")
print(f"Nombre de lignes: {len(df)}")
print(f"Colonnes: {df.columns.tolist()}")
print(f"\nAperçu:")
print(df[['invoice_id', 'customer_id', 'amount', 'days_to_payment']].head(10))
print(f"\nStatistiques days_to_payment:")
print(df['days_to_payment'].describe())