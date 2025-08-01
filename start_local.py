#!/usr/bin/env python3
"""
Script para ejecutar SmartRetail localmente con Docker

Este script automatiza el proceso de construcción y ejecución
de la aplicación SmartRetail usando Docker Compose.

Uso:
    python start_local.py
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_docker():
    """Verificar que Docker esté instalado y funcionando."""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Docker encontrado: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker no está instalado o no está en el PATH")
        print("Por favor, instala Docker Desktop desde: https://www.docker.com/products/docker-desktop")
        return False

def check_docker_compose():
    """Verificar que Docker Compose esté disponible."""
    try:
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Docker Compose encontrado: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker Compose no está disponible")
        return False

def build_and_start():
    """Construir y ejecutar los servicios con Docker Compose."""
    print("\n🔨 Construyendo imágenes de Docker...")
    
    try:
        # Construir las imágenes
        subprocess.run(['docker-compose', 'build'], check=True)
        print("✅ Imágenes construidas exitosamente")
        
        # Ejecutar los servicios
        print("\n🚀 Iniciando servicios...")
        subprocess.run(['docker-compose', 'up', '-d'], check=True)
        print("✅ Servicios iniciados exitosamente")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al construir/ejecutar: {e}")
        return False

def show_status():
    """Mostrar el estado de los servicios."""
    print("\n📊 Estado de los servicios:")
    try:
        subprocess.run(['docker-compose', 'ps'])
    except subprocess.CalledProcessError:
        print("No se pudo obtener el estado de los servicios")

def show_urls():
    """Mostrar las URLs de acceso."""
    print("\n🌐 URLs de acceso:")
    print("=" * 50)
    print("| Servicio                    | URL                    |")
    print("=" * 50)
    print("| Dashboard Principal         | http://localhost:8501  |")
    print("| Detección de Fraude         | http://localhost:8502  |")
    print("| Sistema de Recomendaciones  | http://localhost:8503  |")
    print("| Análisis de Sentimientos    | http://localhost:8504  |")
    print("| Base de Datos (PostgreSQL)  | localhost:5432        |")
    print("| Caché (Redis)              | localhost:6379        |")
    print("=" * 50)

def open_dashboard():
    """Abrir el dashboard principal en el navegador."""
    print("\n🌐 Abriendo dashboard principal...")
    try:
        webbrowser.open('http://localhost:8501')
        print("✅ Dashboard abierto en el navegador")
    except Exception as e:
        print(f"❌ No se pudo abrir el navegador: {e}")

def show_logs():
    """Mostrar logs de los servicios."""
    print("\n📋 Logs de los servicios (últimas 10 líneas):")
    try:
        subprocess.run(['docker-compose', 'logs', '--tail=10'])
    except subprocess.CalledProcessError:
        print("No se pudieron obtener los logs")

def stop_services():
    """Detener todos los servicios."""
    print("\n🛑 Deteniendo servicios...")
    try:
        subprocess.run(['docker-compose', 'down'])
        print("✅ Servicios detenidos")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al detener servicios: {e}")

def main():
    """Función principal."""
    print("🚀 SmartRetail - Iniciador Local")
    print("=" * 50)
    
    # Verificar prerrequisitos
    if not check_docker():
        return
    
    if not check_docker_compose():
        return
    
    # Construir y ejecutar
    if not build_and_start():
        return
    
    # Esperar un poco para que los servicios se inicien
    print("\n⏳ Esperando que los servicios se inicien...")
    time.sleep(10)
    
    # Mostrar información
    show_status()
    show_urls()
    
    # Preguntar si abrir el dashboard
    try:
        response = input("\n¿Deseas abrir el dashboard principal? (s/n): ").lower()
        if response in ['s', 'si', 'sí', 'y', 'yes']:
            open_dashboard()
    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego!")
        return
    
    # Mostrar comandos útiles
    print("\n📚 Comandos útiles:")
    print("- Ver logs: docker-compose logs -f")
    print("- Detener servicios: docker-compose down")
    print("- Reiniciar servicios: docker-compose restart")
    print("- Ver estado: docker-compose ps")
    
    # Mantener el script ejecutándose
    try:
        print("\n🔄 Presiona Ctrl+C para detener los servicios...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Deteniendo servicios...")
        stop_services()
        print("👋 ¡Hasta luego!")

if __name__ == "__main__":
    main() 