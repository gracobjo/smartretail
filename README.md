# 🚀 SmartRetail - Plataforma de Analytics Inteligente

## 📊 Dashboard Funcional - ¡LISTO PARA USAR!

### **🎯 Acceso Rápido:**
```bash
# Método más fácil
python start_dashboard.py

# O directamente
cd sales_analytics_dashboard
python working_dashboard.py
```

**🌐 URL del Dashboard:** http://localhost:8051

---

## 🎯 Descripción del Proyecto

SmartRetail es una plataforma integral de analytics que combina **análisis de ventas**, **detección de fraudes**, **análisis de sentimientos de Twitter** y **dashboards interactivos** para proporcionar insights completos del negocio retail.

## 🏗️ Arquitectura del Sistema

### **Módulos Principales:**

| Módulo | Descripción | Estado | URL |
|--------|-------------|--------|-----|
| 📊 **Dashboard de Ventas** | KPIs y visualizaciones interactivas | ✅ **FUNCIONAL** | http://localhost:8051 |
| 🤖 **Detección de Fraudes** | ML con XGBoost + SHAP | ✅ Completado | - |
| 🐦 **Análisis Twitter** | BERT + PySpark + Streaming | ✅ Completado | - |
| 📈 **Analytics Avanzado** | CLV, Churn, Cohortes | ✅ Completado | - |

## 🚀 Guía de Ejecución Rápida

### **1. Dashboard de Ventas (Recomendado para empezar):**

```bash
# Opción A: Script automático
python start_dashboard.py

# Opción B: Directo
cd sales_analytics_dashboard
python working_dashboard.py

# Opción C: Desde raíz
python sales_analytics_dashboard/working_dashboard.py
```

**✅ Características del Dashboard:**
- 📊 **KPIs en tiempo real** (ventas, clientes, crecimiento)
- 📈 **Gráficos interactivos** con Plotly
- 🎨 **Diseño moderno** y responsive
- ⚡ **Datos simulados** realistas
- 🔄 **Recarga automática** en desarrollo

### **2. Otros Módulos:**

```bash
# Detección de Fraudes
cd fraud_detection
python run_demo.py

# Análisis de Twitter
cd twitter_sentiment_analysis
python run_pipeline.py --mode dashboard

# Notebooks de Jupyter
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 📊 Funcionalidades del Dashboard

### **🎯 KPIs Principales:**
- **💰 Ventas Totales:** Valor acumulado de ventas
- **👥 Clientes Hoy:** Número de clientes del día actual  
- **📈 Crecimiento:** Porcentaje de crecimiento desde el inicio

### **📈 Visualizaciones:**
- **📈 Gráfico de Línea:** Evolución temporal de ventas
- **📊 Gráfico de Barras:** Clientes por día (últimos 30 días)

### **🎨 Características UX:**
- **Responsive:** Se adapta a diferentes pantallas
- **Interactivo:** Hover, zoom, pan en gráficos
- **Moderno:** Paleta de colores profesional
- **Rápido:** Carga instantánea de datos

## 🛠️ Instalación y Configuración

### **Requisitos:**
- Python 3.8+
- pip (gestor de paquetes)

### **Instalación:**
```bash
# Clonar repositorio
git clone https://github.com/gracobjo/smartretail.git
cd smartretail

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar dashboard
python start_dashboard.py
```

### **Verificación:**
```bash
# Verificar Python
python --version

# Verificar Dash
python -c "import dash; print('✅ Dash instalado')"

# Verificar puerto
netstat -an | findstr :8051
```

## 🔧 Solución de Problemas

### **Problemas Comunes:**

| Problema | Solución |
|----------|----------|
| **Puerto 8051 ocupado** | Cambiar puerto en `working_dashboard.py` |
| **Error de Dash** | Usar `app.run()` en lugar de `app.run_server()` |
| **Dependencias faltantes** | `pip install -r requirements.txt` |
| **Archivo no encontrado** | Ejecutar desde directorio correcto |

### **Comandos de Debug:**
```bash
# Verificar archivos
ls sales_analytics_dashboard/

# Verificar puerto
netstat -an | findstr :8051

# Verificar dependencias
python -c "import dash, plotly, pandas; print('✅ Todas las dependencias OK')"
```

## 📁 Estructura del Proyecto

```
SmartRetail/
├── 📊 sales_analytics_dashboard/     # Dashboard principal
│   ├── working_dashboard.py          # ✅ FUNCIONAL
│   ├── start_dashboard.py            # Script de inicio fácil
│   └── src/                          # Módulos avanzados
├── 🤖 fraud_detection/               # Detección de fraudes
├── 🐦 twitter_sentiment_analysis/    # Análisis de Twitter
├── 📈 notebooks/                     # Jupyter notebooks
├── 📚 docs/                          # Documentación
└── 📋 requirements.txt               # Dependencias
```

## 🎯 Casos de Uso

### **Para Ejecutivos:**
- Vista rápida de KPIs principales
- Tendencias de ventas y clientes
- Crecimiento del negocio

### **Para Analistas:**
- Datos detallados en gráficos interactivos
- Exploración de patrones temporales
- Insights para reportes

### **Para Desarrollo:**
- Prototipo de dashboard real
- Base para agregar funcionalidades
- Testing de visualizaciones

## 🔮 Roadmap

### **Próximas Funcionalidades:**
- [ ] **Filtros interactivos** por fecha, región, producto
- [ ] **Drill-down** a datos detallados
- [ ] **Alertas automáticas** para KPIs críticos
- [ ] **Exportación de reportes** en PDF/Excel
- [ ] **Integración con bases de datos** reales
- [ ] **Análisis predictivo** con ML
- [ ] **Autenticación de usuarios**
- [ ] **Múltiples dashboards** por rol

### **Mejoras Técnicas:**
- [ ] **Optimización de rendimiento**
- [ ] **Caché de datos**
- [ ] **API REST** para integración
- [ ] **Docker** para despliegue
- [ ] **CI/CD** automatizado

## 📚 Documentación

### **Documentación Técnica:**
- [📊 Dashboard Documentation](docs/dashboard_documentation.md)
- [🤖 Fraud Detection Guide](fraud_detection/README.md)
- [🐦 Twitter Analysis Guide](twitter_sentiment_analysis/README.md)

### **Notebooks de Ejemplo:**
- [📈 Data Exploration](notebooks/01_data_exploration.ipynb)
- [🔍 Model Comparison](notebooks/02_model_comparison.ipynb)

## 🤝 Contribución

### **Cómo Contribuir:**
1. Fork el repositorio
2. Crear una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abrir un Pull Request

### **Estándares de Código:**
- Usar docstrings en funciones
- Seguir PEP 8 para Python
- Documentar cambios importantes
- Probar funcionalidades antes de commit

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 📞 Contacto

- **GitHub:** [@gracobjo](https://github.com/gracobjo)
- **Proyecto:** [SmartRetail](https://github.com/gracobjo/smartretail)

---

## 🎉 ¡Dashboard Listo!

**🚀 El dashboard está funcionando en:** http://localhost:8051

**📊 Características principales:**
- ✅ KPIs en tiempo real
- ✅ Gráficos interactivos
- ✅ Diseño responsive
- ✅ Fácil de ejecutar
- ✅ Bien documentado

**¡Disfruta explorando los datos de SmartRetail! 🎯** 