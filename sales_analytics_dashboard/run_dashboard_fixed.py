#!/usr/bin/env python3
"""
Script principal para ejecutar el Dashboard de Análisis de Ventas.
"""

import os
import sys
import argparse
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def setup_directories():
    """Crear directorios necesarios."""
    directories = [
        'data/raw',
        'data/processed',
        'data/sample',
        'results/visualizations',
        'results/reports',
        'results/exports',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Directorios creados correctamente!")

def generate_sample_data():
    """Generar datos de ejemplo si no existen."""
    try:
        from src.data.data_generator import DataGenerator
        
        # Verificar si ya existen datos
        if not os.path.exists('data/processed/customers.csv'):
            print("Generando datos de ejemplo...")
            generator = DataGenerator(seed=42)
            customers_df, sales_df, churn_df, cohort_df = generator.generate_sample_data()
            
            print("Datos generados exitosamente!")
            print(f"- Clientes: {len(customers_df)}")
            print(f"- Transacciones: {len(sales_df)}")
            print(f"- Análisis de churn: {len(churn_df)}")
            print(f"- Análisis de cohortes: {len(cohort_df)}")
        else:
            print("Datos de ejemplo ya existen.")
            
    except Exception as e:
        print(f"Error generando datos: {e}")
        return False
    
    return True

def run_dashboard(debug=True, port=8050, host='localhost'):
    """Ejecutar el dashboard."""
    try:
        from src.visualization.dashboard import SalesDashboard
        
        print("Iniciando Sales Analytics Dashboard...")
        print(f"URL: http://{host}:{port}")
        print("Presiona Ctrl+C para detener el servidor")
        
        dashboard = SalesDashboard()
        dashboard.run(debug=debug, port=port)
        
    except Exception as e:
        print(f"Error ejecutando el dashboard: {e}")
        return False
    
    return True

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Sales Analytics Dashboard')
    parser.add_argument('--port', type=int, default=8050, help='Puerto del servidor')
    parser.add_argument('--host', type=str, default='localhost', help='Host del servidor')
    parser.add_argument('--debug', action='store_true', help='Modo debug')
    parser.add_argument('--generate-data', action='store_true', help='Generar datos de ejemplo')
    parser.add_argument('--setup-only', action='store_true', help='Solo configurar directorios')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SALES ANALYTICS DASHBOARD - SmartRetail")
    print("=" * 60)
    
    # Configurar directorios
    setup_directories()
    
    if args.setup_only:
        print("Configuración completada.")
        return
    
    # Generar datos si se solicita
    if args.generate_data:
        if not generate_sample_data():
            print("Error generando datos. Saliendo...")
            return
    
    # Ejecutar dashboard
    if not run_dashboard(debug=args.debug, port=args.port, host=args.host):
        print("Error ejecutando el dashboard. Saliendo...")
        return
    
    print("Dashboard detenido.")

if __name__ == "__main__":
    main() 