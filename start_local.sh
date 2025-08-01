#!/bin/bash

echo "========================================"
echo "   SmartRetail - Iniciador Local"
echo "========================================"
echo

# Verificar si Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "❌ Docker no está instalado o no está en el PATH"
    echo "Por favor, instala Docker desde: https://docs.docker.com/get-docker/"
    exit 1
fi

# Verificar si Docker Compose está disponible
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose no está disponible"
    echo "Por favor, instala Docker Compose desde: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker encontrado: $(docker --version)"
echo "✅ Docker Compose encontrado: $(docker-compose --version)"
echo

echo "🔨 Construyendo imágenes de Docker..."
if ! docker-compose -f docker-compose.yml -f docker-compose.dev.yml build; then
    echo "❌ Error al construir las imágenes"
    exit 1
fi

echo
echo "🚀 Iniciando servicios..."
if ! docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d; then
    echo "❌ Error al iniciar los servicios"
    exit 1
fi

echo
echo "⏳ Esperando que los servicios se inicien..."
sleep 10

echo
echo "📊 Estado de los servicios:"
docker-compose -f docker-compose.yml -f docker-compose.dev.yml ps

echo
echo "🌐 URLs de acceso:"
echo "========================================"
echo "| Servicio                    | URL                    |"
echo "========================================"
echo "| Dashboard Principal         | http://localhost:8501  |"
echo "| Detección de Fraude         | http://localhost:8502  |"
echo "| Sistema de Recomendaciones  | http://localhost:8503  |"
echo "| Análisis de Sentimientos    | http://localhost:8504  |"
echo "| Jupyter Notebook            | http://localhost:8888  |"
echo "| Base de Datos (PostgreSQL)  | localhost:5432        |"
echo "| Caché (Redis)              | localhost:6379        |"
echo "========================================"

echo
read -p "¿Deseas abrir el dashboard principal? (s/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    if command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:8501
    elif command -v open &> /dev/null; then
        open http://localhost:8501
    else
        echo "No se pudo abrir el navegador automáticamente"
        echo "Por favor, abre manualmente: http://localhost:8501"
    fi
fi

echo
echo "📚 Comandos útiles:"
echo "- Ver logs: docker-compose logs -f"
echo "- Detener servicios: docker-compose down"
echo "- Reiniciar servicios: docker-compose restart"
echo "- Ver estado: docker-compose ps"
echo

echo "🔄 Presiona Ctrl+C para detener los servicios..."
echo

# Función para limpiar al salir
cleanup() {
    echo
    echo "🛑 Deteniendo servicios..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
    echo "👋 ¡Hasta luego!"
    exit 0
}

# Capturar Ctrl+C
trap cleanup SIGINT

# Mantener el script ejecutándose
while true; do
    sleep 1
done 