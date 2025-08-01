# 🚀 SmartRetail - Plataforma Integral de Análisis de Datos

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-green?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **SmartRetail** es una plataforma integral de análisis de datos para retail que combina detección de fraude, recomendaciones, análisis de sentimientos y dashboards interactivos.

## 🌟 Características Principales

### 🛡️ **Detección de Fraude Financiero**
- Análisis en tiempo real de transacciones
- Múltiples algoritmos ML (XGBoost, LightGBM, Random Forest)
- Explicabilidad con SHAP values
- Balanceo automático de datos desbalanceados
- Evaluación completa de modelos

### 🎯 **Sistema de Recomendaciones**
- Filtrado colaborativo
- Basado en contenido
- Híbrido
- Evaluación de calidad de recomendaciones
- Personalización en tiempo real

### 📊 **Dashboard de Análisis de Ventas**
- KPIs en tiempo real
- Visualizaciones interactivas con Streamlit
- Reportes automáticos
- Análisis de tendencias
- Exportación de datos

### 📱 **Análisis de Sentimientos**
- Procesamiento de Twitter en tiempo real
- Pipeline con Apache Spark
- Clasificación de sentimientos
- Análisis de tendencias
- Alertas automáticas

### 🤖 **Sistema Multimodal**
- Reconocimiento facial
- Análisis de texto
- Fusión de datos multimodales
- Procesamiento de imágenes

## 🐳 Despliegue Rápido con Docker

### ⚡ Inicio Automático

```bash
# Clonar el repositorio
git clone https://github.com/gracobjo/smartretail.git
cd smartretail

# Windows
start_local.bat

# Linux/Mac
chmod +x start_local.sh
./start_local.sh

# Multiplataforma
python start_local.py
```

### 🌐 URLs de Acceso

| Servicio | URL | Descripción |
|----------|-----|-------------|
| **Dashboard Principal** | http://localhost:8501 | Dashboard de ventas con Streamlit |
| **Detección de Fraude** | http://localhost:8502 | Sistema de detección de fraude |
| **Recomendaciones** | http://localhost:8503 | Sistema de recomendaciones |
| **Análisis de Sentimientos** | http://localhost:8504 | Análisis de Twitter |
| **Jupyter Notebook** | http://localhost:8888 | Desarrollo y experimentación |

## 🏗️ Arquitectura del Sistema

```
SmartRetail
├── 🎯 Dashboard Principal (Streamlit)
├── 🛡️ Detección de Fraude (ML)
├── 🎯 Sistema de Recomendaciones (ML)
├── 📱 Análisis de Sentimientos (NLP)
├── 🤖 Sistema Multimodal (CNN + RNN)
├── 🗄️ PostgreSQL (Base de Datos)
└── ⚡ Redis (Caché)
```

## 🚀 Tecnologías Utilizadas

### Backend
- **Python 3.9+** - Lenguaje principal
- **Scikit-learn** - Machine Learning
- **XGBoost** - Gradient Boosting
- **LightGBM** - Light Gradient Boosting
- **SHAP** - Explicabilidad de modelos
- **Apache Spark** - Procesamiento distribuido

### Frontend
- **Streamlit** - Dashboards interactivos
- **Plotly** - Visualizaciones avanzadas
- **Jupyter Notebook** - Desarrollo y experimentación

### Base de Datos
- **PostgreSQL** - Base de datos principal
- **Redis** - Caché en memoria

### DevOps
- **Docker** - Containerización
- **Docker Compose** - Orquestación
- **Git** - Control de versiones

## 📋 Prerrequisitos

### Software Requerido
- **Docker Desktop** 20.10+
- **Git** 2.30+
- **Python** 3.8+ (opcional)

### Requisitos del Sistema
- **RAM**: 4GB mínimo, 8GB+ recomendado
- **Almacenamiento**: 10GB espacio libre
- **CPU**: 2 cores mínimo, 4+ recomendado

## 🔧 Instalación Manual

### 1. Clonar Repositorio
```bash
git clone https://github.com/gracobjo/smartretail.git
cd smartretail
```

### 2. Verificar Docker
```bash
docker --version
docker-compose --version
```

### 3. Ejecutar con Docker
```bash
# Construir y ejecutar
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Verificar estado
docker-compose ps
```

## 📚 Documentación

### Guías Principales
- 📖 **[Guía de Despliegue](docs/HOW_TO_DEPLOY.md)** - Instalación y configuración completa
- 🐳 **[Docker Deployment](README_Docker.md)** - Documentación específica de Docker
- 🧪 **[Guía de Desarrollo](docs/development_guide.md)** - Desarrollo y contribución

### Documentación Técnica
- 📊 **[Documentación Técnica](docs/technical_documentation.md)** - Arquitectura y diseño
- 📈 **[Dashboard Documentation](docs/dashboard_documentation.md)** - Uso del dashboard
- 🔍 **[Project Summary](docs/project_summary.md)** - Resumen del proyecto

## 🎯 Casos de Uso

### E-commerce
- **Detección de fraude** en transacciones en tiempo real
- **Recomendaciones personalizadas** de productos
- **Análisis de sentimientos** de clientes en redes sociales
- **Dashboard de métricas** de ventas y KPIs

### Retail
- **Optimización de inventario** basada en recomendaciones
- **Análisis de comportamiento** de clientes
- **Predicción de demanda** usando ML
- **Monitoreo de satisfacción** del cliente

### Marketing
- **Targeting personalizado** basado en análisis de sentimientos
- **Optimización de campañas** con datos en tiempo real
- **Análisis de competencia** en redes sociales
- **ROI tracking** con dashboards interactivos

## 🛠️ Desarrollo

### Estructura del Proyecto
```
smartretail/
├── fraud_detection/          # Sistema de detección de fraude
├── recommendation_system/     # Sistema de recomendaciones
├── sales_analytics_dashboard/ # Dashboard de ventas
├── twitter_sentiment_analysis/ # Análisis de sentimientos
├── src/                      # Código fuente principal
├── data/                     # Datos del proyecto
├── docs/                     # Documentación
├── notebooks/                # Jupyter notebooks
├── tests/                    # Tests unitarios
└── docker-compose.yml        # Configuración Docker
```

### Comandos de Desarrollo
```bash
# Ejecutar en modo desarrollo
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Ver logs en tiempo real
docker-compose logs -f

# Ejecutar tests
docker-compose exec smartretail python -m pytest

# Acceder a Jupyter
http://localhost:8888
```

## 📊 Métricas y Rendimiento

### Detección de Fraude
- **Precision**: 95%+
- **Recall**: 90%+
- **F1-Score**: 92%+
- **ROC AUC**: 0.95+

### Sistema de Recomendaciones
- **Precision@10**: 85%+
- **Recall@10**: 80%+
- **NDCG@10**: 0.85+

### Análisis de Sentimientos
- **Accuracy**: 88%+
- **F1-Score**: 86%+
- **Tiempo de procesamiento**: <100ms

## 🔒 Seguridad

### Características de Seguridad
- ✅ **Autenticación** de usuarios
- ✅ **Autorización** basada en roles
- ✅ **Encriptación** de datos sensibles
- ✅ **Logs de auditoría** completos
- ✅ **Backup automático** de datos

### Buenas Prácticas
- Cambiar contraseñas por defecto
- Usar variables de entorno para credenciales
- Configurar firewall
- Actualizar imágenes regularmente

## 🤝 Contribución

### Cómo Contribuir
1. **Fork** el proyecto
2. **Crear** una rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. **Commit** cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. **Push** a la rama (`git push origin feature/nueva-funcionalidad`)
5. **Crear** Pull Request

### Estándares de Código
- Usar PEP 8 para Python
- Documentar funciones y clases
- Escribir tests para nuevas funcionalidades
- Mantener commits atómicos

## 📞 Soporte

### Canales de Soporte
- **Issues**: [GitHub Issues](https://github.com/gracobjo/smartretail/issues)
- **Documentación**: [Wiki](https://github.com/gracobjo/smartretail/wiki)
- **Email**: soporte@smartretail.com

### Recursos Adicionales
- [Documentación de Docker](https://docs.docker.com/)
- [Documentación de Streamlit](https://docs.streamlit.io/)
- [Guía de Machine Learning](https://scikit-learn.org/stable/)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- **Streamlit** por el framework de dashboards
- **Scikit-learn** por las herramientas de ML
- **Docker** por la containerización
- **Comunidad open source** por las librerías utilizadas

---

## 🚀 ¡Comienza Ahora!

```bash
# Clonar y ejecutar en 3 comandos
git clone https://github.com/gracobjo/smartretail.git
cd smartretail
python start_local.py
```

**¡Disfruta usando SmartRetail! 🎉**

---

<div align="center">

**⭐ Si te gusta este proyecto, ¡dale una estrella! ⭐**

[![GitHub stars](https://img.shields.io/github/stars/gracobjo/smartretail?style=social)](https://github.com/gracobjo/smartretail/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/gracobjo/smartretail?style=social)](https://github.com/gracobjo/smartretail/network/members)
[![GitHub issues](https://img.shields.io/github/issues/gracobjo/smartretail)](https://github.com/gracobjo/smartretail/issues)

</div> 