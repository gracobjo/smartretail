#!/usr/bin/env python3
"""
Dashboard de prueba muy simple para SmartRetail.
"""

import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import numpy as np

def create_test_dashboard():
    """Crear dashboard de prueba."""
    app = dash.Dash(__name__)
    
    # Datos de ejemplo
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    sales_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.randn(len(dates)).cumsum() + 1000,
        'customers': np.random.randint(50, 200, len(dates))
    })
    
    # Layout simple
    app.layout = html.Div([
        html.H1("🚀 SmartRetail - Dashboard de Prueba", 
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
    
    return app

def main():
    """Función principal."""
    print("=" * 60)
    print("SMARTRETAIL - DASHBOARD DE PRUEBA")
    print("=" * 60)
    print("🚀 Iniciando dashboard...")
    print("📊 URL: http://localhost:8050")
    print("⏹️  Presiona Ctrl+C para detener")
    print("=" * 60)
    
    app = create_test_dashboard()
    app.run_server(debug=True, port=8050, host='0.0.0.0')

if __name__ == "__main__":
    main() 