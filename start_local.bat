@echo off
echo ========================================
echo    SmartRetail - Iniciador Local
echo ========================================
echo.

REM Verificar si Docker está instalado
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker no está instalado o no está en el PATH
    echo Por favor, instala Docker Desktop desde: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Verificar si Docker Compose está disponible
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose no está disponible
    pause
    exit /b 1
)

echo ✅ Docker encontrado
echo ✅ Docker Compose encontrado
echo.

echo 🔨 Construyendo imágenes de Docker...
docker-compose -f docker-compose.yml -f docker-compose.dev.yml build
if %errorlevel% neq 0 (
    echo ❌ Error al construir las imágenes
    pause
    exit /b 1
)

echo.
echo 🚀 Iniciando servicios...
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
if %errorlevel% neq 0 (
    echo ❌ Error al iniciar los servicios
    pause
    exit /b 1
)

echo.
echo ⏳ Esperando que los servicios se inicien...
timeout /t 10 /nobreak >nul

echo.
echo 📊 Estado de los servicios:
docker-compose -f docker-compose.yml -f docker-compose.dev.yml ps

echo.
echo 🌐 URLs de acceso:
echo ========================================
echo | Servicio                    | URL                    |
echo ========================================
echo | Dashboard Principal         | http://localhost:8501  |
echo | Detección de Fraude         | http://localhost:8502  |
echo | Sistema de Recomendaciones  | http://localhost:8503  |
echo | Análisis de Sentimientos    | http://localhost:8504  |
echo | Jupyter Notebook            | http://localhost:8888  |
echo | Base de Datos (PostgreSQL)  | localhost:5432        |
echo | Caché (Redis)              | localhost:6379        |
echo ========================================

echo.
set /p open_dashboard="¿Deseas abrir el dashboard principal? (s/n): "
if /i "%open_dashboard%"=="s" (
    start http://localhost:8501
)

echo.
echo 📚 Comandos útiles:
echo - Ver logs: docker-compose logs -f
echo - Detener servicios: docker-compose down
echo - Reiniciar servicios: docker-compose restart
echo - Ver estado: docker-compose ps
echo.

echo 🔄 Presiona Ctrl+C para detener los servicios...
echo.
pause 