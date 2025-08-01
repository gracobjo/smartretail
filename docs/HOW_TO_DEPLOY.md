# 🚀 SmartRetail - Guía de Despliegue Local

## 📋 Tabla de Contenidos

1. [Prerrequisitos](#prerrequisitos)
2. [Instalación Rápida](#instalación-rápida)
3. [Despliegue Manual](#despliegue-manual)
4. [Configuración Avanzada](#configuración-avanzada)
5. [Troubleshooting](#troubleshooting)
6. [Comandos Útiles](#comandos-útiles)
7. [Arquitectura del Sistema](#arquitectura-del-sistema)
8. [Desarrollo](#desarrollo)

---

## 🔧 Prerrequisitos

### Software Requerido

| Software | Versión Mínima | Descarga |
|----------|----------------|----------|
| **Docker Desktop** | 20.10+ | [Docker Desktop](https://www.docker.com/products/docker-desktop) |
| **Git** | 2.30+ | [Git](https://git-scm.com/downloads) |
| **Python** | 3.8+ | [Python](https://www.python.org/downloads/) |

### Verificar Instalación

```bash
# Verificar Docker
docker --version

# Verificar Docker Compose
docker-compose --version

# Verificar Git
git --version

# Verificar Python
python --version
```

### Requisitos del Sistema

- **RAM**: Mínimo 4GB, recomendado 8GB+
- **Almacenamiento**: 10GB de espacio libre
- **CPU**: 2 cores mínimo, 4+ recomendado
- **Sistema Operativo**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

---

## ⚡ Instalación Rápida

### Opción 1: Script Automático (Recomendado)

#### Windows
```bash
# Clonar el repositorio
git clone https://github.com/gracobjo/smartretail.git
cd smartretail

# Ejecutar script de inicio
start_local.bat
```

#### Linux/Mac
```bash
# Clonar el repositorio
git clone https://github.com/gracobjo/smartretail.git
cd smartretail

# Hacer ejecutable y ejecutar
chmod +x start_local.sh
./start_local.sh
```

#### Multiplataforma (Python)
```bash
# Clonar el repositorio
git clone https://github.com/gracobjo/smartretail.git
cd smartretail

# Ejecutar script Python
python start_local.py
```

### Opción 2: Comandos Manuales

```bash
# 1. Clonar repositorio
git clone https://github.com/gracobjo/smartretail.git
cd smartretail

# 2. Construir imágenes Docker
docker-compose -f docker-compose.yml -f docker-compose.dev.yml build

# 3. Ejecutar servicios
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# 4. Verificar estado
docker-compose ps
```

---

## 🌐 Acceso a los Servicios

Una vez ejecutado, podrás acceder a los siguientes servicios:

| Servicio | URL | Descripción | Estado |
|----------|-----|-------------|--------|
| **Dashboard Principal** | http://localhost:8501 | Dashboard de ventas con Streamlit | ✅ Activo |
| **Detección de Fraude** | http://localhost:8502 | Sistema de detección de fraude | ✅ Activo |
| **Recomendaciones** | http://localhost:8503 | Sistema de recomendaciones | ✅ Activo |
| **Análisis de Sentimientos** | http://localhost:8504 | Análisis de Twitter | ✅ Activo |
| **Jupyter Notebook** | http://localhost:8888 | Desarrollo y experimentación | ✅ Activo |
| **PostgreSQL** | localhost:5432 | Base de datos | ✅ Activo |
| **Redis** | localhost:6379 | Caché | ✅ Activo |

---

## 🔧 Despliegue Manual

### Paso 1: Preparar el Entorno

```bash
# Crear directorios necesarios
mkdir -p data results logs
mkdir -p fraud_detection/{data,results,models,logs}
mkdir -p recommendation_system/{data,results}
mkdir -p sales_analytics_dashboard/{data,results}
mkdir -p twitter_sentiment_analysis/{data,results}
```

### Paso 2: Configurar Variables de Entorno

Crear archivo `.env`:
```env
# Configuración del entorno
ENVIRONMENT=development
DEBUG=true
PYTHONPATH=/app

# Base de datos
POSTGRES_DB=smartretail_dev
POSTGRES_USER=smartretail_user
POSTGRES_PASSWORD=smartretail_pass_dev

# Redis
REDIS_URL=redis://localhost:6379

# APIs (opcional)
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
```

### Paso 3: Construir Imágenes

```bash
# Construir imagen base
docker build -t smartretail:latest .

# Verificar imagen creada
docker images | grep smartretail
```

### Paso 4: Ejecutar Servicios

```bash
# Ejecutar todos los servicios
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Ejecutar solo servicios específicos
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d smartretail fraud-detection

# Ejecutar en modo foreground (ver logs)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

---

## ⚙️ Configuración Avanzada

### Personalizar Puertos

Editar `docker-compose.dev.yml`:
```yaml
services:
  smartretail:
    ports:
      - "8501:8501"  # Cambiar 8501 por el puerto deseado
```

### Configurar Recursos

```yaml
services:
  smartretail:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
```

### Configurar Volúmenes

```yaml
services:
  smartretail:
    volumes:
      - ./data:/app/data:rw
      - ./results:/app/results:rw
      - ./logs:/app/logs:rw
```

### Configurar Redes

```yaml
networks:
  smartretail_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

services:
  smartretail:
    networks:
      - smartretail_network
```

---

## 🚨 Troubleshooting

### Problemas Comunes

#### 1. Puerto ya en uso
```bash
# Verificar puertos en uso
netstat -an | findstr :8501  # Windows
lsof -i :8501                # Linux/Mac

# Cambiar puerto en docker-compose.dev.yml
ports:
  - "8505:8501"  # Usar puerto 8505 en lugar de 8501
```

#### 2. Error de permisos
```bash
# Linux/Mac
sudo chown -R $USER:$USER ./data ./results ./logs

# Windows (PowerShell como administrador)
icacls . /grant Everyone:F /T
```

#### 3. Memoria insuficiente
```bash
# Aumentar memoria en Docker Desktop
# Settings > Resources > Memory: 8GB

# O limitar memoria en docker-compose.dev.yml
deploy:
  resources:
    limits:
      memory: 4G
```

#### 4. Error de dependencias
```bash
# Reconstruir imagen sin caché
docker-compose build --no-cache

# Limpiar imágenes Docker
docker system prune -a
```

#### 5. Servicios no inician
```bash
# Ver logs de servicios
docker-compose logs smartretail
docker-compose logs fraud-detection

# Reiniciar servicios
docker-compose restart

# Verificar estado
docker-compose ps
```

### Logs de Diagnóstico

```bash
# Ver logs en tiempo real
docker-compose logs -f

# Ver logs de servicio específico
docker-compose logs -f smartretail

# Ver últimas 50 líneas
docker-compose logs --tail=50

# Ver logs con timestamps
docker-compose logs -f -t
```

---

## 📚 Comandos Útiles

### Gestión de Servicios

```bash
# Iniciar servicios
docker-compose up -d

# Detener servicios
docker-compose down

# Reiniciar servicios
docker-compose restart

# Ver estado
docker-compose ps

# Ver logs
docker-compose logs -f
```

### Gestión de Imágenes

```bash
# Construir imágenes
docker-compose build

# Reconstruir sin caché
docker-compose build --no-cache

# Ver imágenes
docker images

# Eliminar imágenes
docker rmi smartretail:latest
```

### Gestión de Contenedores

```bash
# Ver contenedores
docker ps

# Ver todos los contenedores
docker ps -a

# Ejecutar comando en contenedor
docker-compose exec smartretail bash

# Ver logs de contenedor
docker logs smartretail-main-dev
```

### Limpieza

```bash
# Detener y eliminar contenedores
docker-compose down

# Eliminar volúmenes
docker-compose down -v

# Limpiar todo
docker system prune -a

# Eliminar imágenes no utilizadas
docker image prune -a
```

---

## 🏗️ Arquitectura del Sistema

### Servicios Principales

```
SmartRetail
├── smartretail (Dashboard Principal)
│   ├── Puerto: 8501
│   ├── Tecnología: Streamlit
│   └── Función: Dashboard de ventas
├── fraud-detection (Detección de Fraude)
│   ├── Puerto: 8502
│   ├── Tecnología: Python + ML
│   └── Función: Detección de fraude
├── recommendation-system (Recomendaciones)
│   ├── Puerto: 8503
│   ├── Tecnología: Python + ML
│   └── Función: Sistema de recomendaciones
├── sentiment-analysis (Análisis de Sentimientos)
│   ├── Puerto: 8504
│   ├── Tecnología: Python + NLP
│   └── Función: Análisis de Twitter
├── jupyter (Desarrollo)
│   ├── Puerto: 8888
│   ├── Tecnología: Jupyter Notebook
│   └── Función: Desarrollo y experimentación
├── postgres (Base de Datos)
│   ├── Puerto: 5432
│   ├── Tecnología: PostgreSQL
│   └── Función: Almacenamiento de datos
└── redis (Caché)
    ├── Puerto: 6379
    ├── Tecnología: Redis
    └── Función: Caché en memoria
```

### Volúmenes de Datos

```
Volúmenes Persistentes
├── ./data
│   ├── Datos del proyecto
│   └── Datasets de entrenamiento
├── ./results
│   ├── Resultados de análisis
│   └── Modelos entrenados
├── ./logs
│   ├── Logs de aplicación
│   └── Logs de errores
└── postgres_data_dev
    └── Base de datos PostgreSQL
```

---

## 🧪 Desarrollo

### Modo Desarrollo

```bash
# Ejecutar en modo desarrollo
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Hot reload activado
# Los cambios en el código se reflejan automáticamente
```

### Desarrollo con Jupyter

```bash
# Acceder a Jupyter Notebook
http://localhost:8888

# Crear nuevo notebook
# File > New > Python 3
```

### Debugging

```bash
# Ejecutar en modo debug
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Ver logs en tiempo real
docker-compose logs -f smartretail
```

### Testing

```bash
# Ejecutar tests
docker-compose exec smartretail python -m pytest

# Ejecutar tests específicos
docker-compose exec smartretail python -m pytest tests/test_fraud_detection.py
```

---

## 📊 Monitoreo

### Métricas del Sistema

```bash
# Ver uso de recursos
docker stats

# Ver información de contenedores
docker-compose ps

# Ver logs de todos los servicios
docker-compose logs -f
```

### Alertas

```bash
# Verificar salud de servicios
curl http://localhost:8501/health

# Verificar conectividad de base de datos
docker-compose exec postgres pg_isready
```

---

## 🔒 Seguridad

### Buenas Prácticas

1. **Cambiar contraseñas por defecto**
2. **Usar variables de entorno para credenciales**
3. **Configurar firewall**
4. **Actualizar imágenes regularmente**
5. **Usar secrets para datos sensibles**

### Variables de Entorno Seguras

```bash
# Crear archivo .env.secret
echo "POSTGRES_PASSWORD=tu_password_seguro" > .env.secret
echo "REDIS_PASSWORD=tu_redis_password" >> .env.secret
```

---

## 📈 Escalabilidad

### Escalar Servicios

```bash
# Escalar dashboard a 3 instancias
docker-compose up -d --scale smartretail=3

# Escalar todos los servicios
docker-compose up -d --scale smartretail=2 --scale fraud-detection=2
```

### Load Balancer

```bash
# Agregar nginx como load balancer
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## 🤝 Contribución

### Flujo de Desarrollo

1. **Fork el proyecto**
2. **Crear rama feature**
   ```bash
   git checkout -b feature/nueva-funcionalidad
   ```
3. **Hacer cambios**
4. **Commit cambios**
   ```bash
   git commit -am 'Agregar nueva funcionalidad'
   ```
5. **Push a la rama**
   ```bash
   git push origin feature/nueva-funcionalidad
   ```
6. **Crear Pull Request**

### Estándares de Código

- Usar PEP 8 para Python
- Documentar funciones y clases
- Escribir tests para nuevas funcionalidades
- Mantener commits atómicos

---

## 📞 Soporte

### Canales de Soporte

- **Issues**: [GitHub Issues](https://github.com/gracobjo/smartretail/issues)
- **Documentación**: [Wiki](https://github.com/gracobjo/smartretail/wiki)
- **Email**: soporte@smartretail.com

### Recursos Adicionales

- [Documentación de Docker](https://docs.docker.com/)
- [Documentación de Docker Compose](https://docs.docker.com/compose/)
- [Documentación de Streamlit](https://docs.streamlit.io/)
- [Guía de Machine Learning](https://scikit-learn.org/stable/)

---

## 📝 Changelog

### Versión 1.0.0
- ✅ Configuración inicial de Docker
- ✅ Scripts de inicio automático
- ✅ Documentación completa
- ✅ Configuración de desarrollo
- ✅ Troubleshooting guide

---

**¡Disfruta usando SmartRetail! 🚀** 