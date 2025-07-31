#!/usr/bin/env python3
"""
Script de inicio fácil para el Dashboard SmartRetail.
"""

import os
import sys
import subprocess
import webbrowser
import time

def main():
    """Función principal."""
    print("=" * 60)
    print("🚀 SMART RETAIL - INICIADOR DE DASHBOARD")
    print("=" * 60)
    
    # Cambiar al directorio correcto
    dashboard_dir = "sales_analytics_dashboard"
    if not os.path.exists(dashboard_dir):
        print(f"❌ Error: No se encuentra el directorio {dashboard_dir}")
        return
    
    os.chdir(dashboard_dir)
    print(f"✅ Directorio cambiado a: {os.getcwd()}")
    
    # Verificar que el archivo existe
    dashboard_file = "working_dashboard.py"
    if not os.path.exists(dashboard_file):
        print(f"❌ Error: No se encuentra el archivo {dashboard_file}")
        return
    
    print("🚀 Iniciando dashboard...")
    print("📊 URL: http://localhost:8051")
    print("🌐 Se abrirá automáticamente en tu navegador")
    print("⏹️  Presiona Ctrl+C para detener")
    print("=" * 60)
    
    # Esperar un momento y abrir el navegador
    time.sleep(3)
    try:
        webbrowser.open('http://localhost:8051')
        print("✅ Navegador abierto automáticamente")
    except:
        print("⚠️  No se pudo abrir el navegador automáticamente")
        print("📊 Abre manualmente: http://localhost:8051")
    
    # Ejecutar el dashboard
    try:
        subprocess.run([sys.executable, dashboard_file], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard detenido por el usuario")
    except Exception as e:
        print(f"❌ Error ejecutando el dashboard: {e}")

if __name__ == "__main__":
    main() 