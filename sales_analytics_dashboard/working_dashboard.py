#!/usr/bin/env python3
"""
Dashboard funcional para SmartRetail.
"""

import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import numpy as np

# Crear datos de ejemplo
np.random.seed(42)
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
sales_data = pd.DataFrame({
    'date': dates,
    'sales': np.random.randn(len(dates)).cumsum() + 1000,
    'customers': np.random.randint(50, 200, len(dates))
})

# Crear aplicación Dash
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("🚀 SmartRetail - Dashboard Funcional", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    html.Div([
        html.Div([
            html.H3("💰 Ventas Totales"),
            html.H2(f"${sales_data['sales'].iloc[-1]:,.0f}")
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'margin': '10px'}),
        
        html.Div([
            html.H3("👥 Clientes Hoy"),
            html.H2(f"{sales_data['customers'].iloc[-1]:,}")
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'margin': '10px'}),
        
        html.Div([
            html.H3("📈 Crecimiento"),
            html.H2(f"+{((sales_data['sales'].iloc[-1] / sales_data['sales'].iloc[0] - 1) * 100):.1f}%")
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'margin': '10px'})
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': 30}),
    
    html.Div([
        dcc.Graph(
            id='sales-chart',
            figure=px.line(
                sales_data, 
                x='date', 
                y='sales',
                title='Evolución de Ventas - SmartRetail'
            )
        )
    ]),
    
    html.Div([
        dcc.Graph(
            id='customers-chart',
            figure=px.bar(
                sales_data.tail(30), 
                x='date', 
                y='customers',
                title='Clientes por Día (Últimos 30 días)'
            )
        )
    ])
])

if __name__ == '__main__':
    print("=" * 60)
    print("SMARTRETAIL - DASHBOARD FUNCIONAL")
    print("=" * 60)
    print("🚀 Iniciando dashboard...")
    print("📊 URL: http://localhost:8051")
    print("⏹️  Presiona Ctrl+C para detener")
    print("=" * 60)
    
    try:
        app.run(debug=True, port=8051, host='127.0.0.1')
    except Exception as e:
        print(f"Error: {e}")
        print("Intentando con puerto 8052...")
        app.run(debug=True, port=8052, host='127.0.0.1') 