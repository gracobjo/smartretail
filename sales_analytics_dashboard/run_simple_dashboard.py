#!/usr/bin/env python3
"""
Versión simplificada del Dashboard de Análisis de Ventas.
"""

import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta
import os

def generate_simple_data():
    """Generar datos simples para el dashboard."""
    print("Generando datos simples...")
    
    # Generar datos de clientes
    np.random.seed(42)
    n_customers = 1000
    n_transactions = 5000
    
    # Clientes
    customers_data = []
    for i in range(n_customers):
        customer = {
            'customer_id': f'CUST_{i:06d}',
            'segment': np.random.choice(['Premium', 'Regular', 'Occasional']),
            'region': np.random.choice(['North', 'South', 'East', 'West']),
            'acquisition_date': datetime.now() - timedelta(days=np.random.randint(1, 365))
        }
        customers_data.append(customer)
    
    customers_df = pd.DataFrame(customers_data)
    
    # Transacciones
    transactions_data = []
    products = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports']
    
    for i in range(n_transactions):
        customer = np.random.choice(customers_df['customer_id'])
        product = np.random.choice(products)
        price = np.random.uniform(20, 500)
        quantity = np.random.poisson(1.5) + 1
        
        transaction = {
            'transaction_id': f'TXN_{i:06d}',
            'customer_id': customer,
            'product_category': product,
            'quantity': quantity,
            'unit_price': round(price, 2),
            'total_amount': round(price * quantity, 2),
            'transaction_date': datetime.now() - timedelta(days=np.random.randint(1, 365)),
            'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'Cash'])
        }
        transactions_data.append(transaction)
    
    transactions_df = pd.DataFrame(transactions_data)
    
    # Guardar datos
    os.makedirs('data/processed', exist_ok=True)
    customers_df.to_csv('data/processed/customers.csv', index=False)
    transactions_df.to_csv('data/processed/sales.csv', index=False)
    
    print(f"✅ Datos generados: {len(customers_df)} clientes, {len(transactions_df)} transacciones")
    return customers_df, transactions_df

def create_dashboard():
    """Crear dashboard simple."""
    app = dash.Dash(__name__)
    
    # Cargar datos
    try:
        customers_df = pd.read_csv('data/processed/customers.csv')
        sales_df = pd.read_csv('data/processed/sales.csv')
        sales_df['transaction_date'] = pd.to_datetime(sales_df['transaction_date'])
    except:
        print("Generando datos...")
        customers_df, sales_df = generate_simple_data()
        sales_df['transaction_date'] = pd.to_datetime(sales_df['transaction_date'])
    
    # Layout del dashboard
    app.layout = html.Div([
        html.H1("📊 SmartRetail - Dashboard de Ventas", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
        
        # KPIs principales
        html.Div([
            html.Div([
                html.H3("💰 Ingresos Totales"),
                html.H2(f"${sales_df['total_amount'].sum():,.0f}")
            ], className="kpi-card"),
            html.Div([
                html.H3("👥 Clientes"),
                html.H2(f"{len(customers_df):,}")
            ], className="kpi-card"),
            html.Div([
                html.H3("🛍️ Transacciones"),
                html.H2(f"{len(sales_df):,}")
            ], className="kpi-card"),
            html.Div([
                html.H3("📈 Promedio"),
                html.H2(f"${sales_df['total_amount'].mean():.0f}")
            ], className="kpi-card")
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': 30}),
        
        # Gráficos
        html.Div([
            html.Div([
                dcc.Graph(
                    id='sales-trend',
                    figure=px.line(
                        sales_df.groupby(sales_df['transaction_date'].dt.to_period('M'))['total_amount'].sum().reset_index(),
                        x='transaction_date',
                        y='total_amount',
                        title='Evolución de Ventas Mensuales'
                    )
                )
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(
                    id='category-sales',
                    figure=px.pie(
                        sales_df.groupby('product_category')['total_amount'].sum().reset_index(),
                        values='total_amount',
                        names='product_category',
                        title='Ventas por Categoría'
                    )
                )
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        html.Div([
            html.Div([
                dcc.Graph(
                    id='segment-distribution',
                    figure=px.bar(
                        customers_df['segment'].value_counts().reset_index(),
                        x='index',
                        y='segment',
                        title='Distribución de Segmentos'
                    )
                )
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(
                    id='region-sales',
                    figure=px.bar(
                        sales_df.merge(customers_df[['customer_id', 'region']], on='customer_id')
                        .groupby('region')['total_amount'].sum().reset_index(),
                        x='region',
                        y='total_amount',
                        title='Ventas por Región'
                    )
                )
            ], style={'width': '50%', 'display': 'inline-block'})
        ])
    ])
    
    return app

def main():
    """Función principal."""
    print("=" * 60)
    print("SMARTRETAIL - DASHBOARD SIMPLIFICADO")
    print("=" * 60)
    
    # Crear directorios
    os.makedirs('data/processed', exist_ok=True)
    
    # Generar datos si no existen
    if not os.path.exists('data/processed/customers.csv'):
        generate_simple_data()
    
    # Crear y ejecutar dashboard
    app = create_dashboard()
    
    print("\n🚀 Iniciando dashboard...")
    print("📊 URL: http://localhost:8050")
    print("⏹️  Presiona Ctrl+C para detener")
    print("=" * 60)
    
    app.run_server(debug=True, port=8050)

if __name__ == "__main__":
    main() 