# 🚀 SmartRetail - Plataforma Integral de Analytics

## 📋 Descripción General

SmartRetail es una plataforma completa de analytics que integra múltiples sistemas de análisis de datos para el sector retail. Incluye detección de fraude, análisis de sentimientos en redes sociales, sistemas de recomendación, análisis de ventas y dashboards interactivos.

## 🎯 **Sistemas Integrados**

### 🛡️ **Detección de Fraude**
- **Ubicación:** `fraud_detection/`
- **Descripción:** Sistema de machine learning para detectar transacciones fraudulentas
- **Tecnologías:** XGBoost, Random Forest, SHAP, SMOTE
- **Estado:** ✅ **Funcionando**
- **Documentación:** [Ver documentación completa](fraud_detection/README.md)

### 📊 **Análisis de Sentimientos en Twitter**
- **Ubicación:** `twitter_sentiment_analysis/`
- **Descripción:** Pipeline completo para análisis de sentimientos en tiempo real
- **Tecnologías:** BERT, PySpark, Dash, Hugging Face
- **Estado:** ✅ **Funcionando**
- **Documentación:** [Ver documentación completa](twitter_sentiment_analysis/README.md)

### 🎯 **Sistema de Recomendaciones**
- **Ubicación:** `recommendation_system/`
- **Descripción:** Sistema híbrido de recomendaciones (colaborativo + basado en contenido)
- **Tecnologías:** Scikit-learn, Pandas, NumPy
- **Estado:** ✅ **Funcionando**
- **Documentación:** [Ver documentación completa](recommendation_system/README.md)

### 📈 **Dashboard de Análisis de Ventas**
- **Ubicación:** `sales_analytics_dashboard/`
- **Descripción:** Dashboard interactivo para análisis de ventas y KPIs
- **Tecnologías:** Dash, Plotly, Pandas
- **Estado:** ✅ **Funcionando**
- **Documentación:** [Ver documentación completa](sales_analytics_dashboard/README.md)

## 📚 **Documentación**

### 📊 **Plantillas de Documentación**
- **[Plantilla de Dashboard](docs/dashboard_template.md)** - Plantilla completa para documentar dashboards de negocio con Power BI o Tableau
- **[Ejemplo Práctico](docs/dashboard_example.md)** - Ejemplo real de aplicación de la plantilla para un dashboard de ventas

### 🛡️ **Documentación de Sistemas**
- **[Detección de Fraude](fraud_detection/docs/)** - Documentación técnica y funcional del sistema de fraude
- **[Análisis de Twitter](twitter_sentiment_analysis/docs/)** - Documentación completa del sistema de sentimientos
- **[Recomendaciones](recommendation_system/docs/)** - Guías del sistema de recomendaciones
- **[Dashboard de Ventas](sales_analytics_dashboard/docs/)** - Documentación del dashboard de analytics

### 📋 **Guías de Uso**
- **[Guía de Inicio Rápido](docs/quick_start_guide.md)** - Cómo comenzar con SmartRetail
- **[Guía de Instalación](docs/installation_guide.md)** - Instalación y configuración
- **[Guía de Troubleshooting](docs/troubleshooting_guide.md)** - Solución de problemas comunes

## 🚀 **Inicio Rápido**

### **Prerrequisitos**
```bash
# Python 3.8+
python --version

# Git
git --version

# Dependencias del sistema
pip install -r requirements.txt
```

### **Instalación**
```bash
# Clonar el repositorio
git clone https://github.com/gracobjo/smartretail.git
cd smartretail

# Instalar dependencias
pip install -r requirements.txt
```

### **Ejecución de Sistemas**

#### **Detección de Fraude**
```bash
cd fraud_detection
python run_demo.py
```

#### **Análisis de Twitter**
```bash
cd twitter_sentiment_analysis
python run_pipeline.py --mode dashboard
```

#### **Sistema de Recomendaciones**
```bash
cd recommendation_system
python main.py
```

#### **Dashboard de Ventas**
```bash
cd sales_analytics_dashboard
python run_dashboard.py
```

## 📊 **Estructura del Proyecto**

```
SmartRetail/
├── 📊 docs/                           # Documentación general
│   ├── dashboard_template.md          # Plantilla para dashboards
│   ├── dashboard_example.md           # Ejemplo práctico
│   └── ...
├── 🛡️ fraud_detection/               # Sistema de detección de fraude
│   ├── src/                          # Código fuente
│   ├── docs/                         # Documentación específica
│   ├── results/                      # Resultados y visualizaciones
│   └── run_demo.py                   # Script principal
├── 📊 twitter_sentiment_analysis/     # Análisis de sentimientos
│   ├── src/                          # Código fuente
│   ├── docs/                         # Documentación
│   └── run_pipeline.py               # Script principal
├── 🎯 recommendation_system/          # Sistema de recomendaciones
│   ├── src/                          # Código fuente
│   ├── docs/                         # Documentación
│   └── main.py                       # Script principal
├── 📈 sales_analytics_dashboard/      # Dashboard de ventas
│   ├── src/                          # Código fuente
│   ├── docs/                         # Documentación
│   └── run_dashboard.py              # Script principal
└── 📋 requirements.txt                # Dependencias del proyecto
```

## 🎯 **Casos de Uso**

### **Para Equipos de Negocio**
- **Monitoreo de KPIs** en tiempo real
- **Detección de fraudes** en transacciones
- **Análisis de sentimientos** de clientes
- **Recomendaciones personalizadas** para clientes

### **Para Equipos Técnicos**
- **Desarrollo de modelos** de machine learning
- **Integración de datos** de múltiples fuentes
- **Optimización de algoritmos** de detección
- **Escalabilidad** de sistemas de analytics

### **Para Analistas de Datos**
- **Exploración de datos** con visualizaciones
- **Análisis predictivo** de tendencias
- **Identificación de patrones** en comportamiento
- **Generación de insights** accionables

## 🛠️ **Tecnologías Utilizadas**

### **Machine Learning**
- **XGBoost** - Gradient boosting para clasificación
- **Random Forest** - Ensemble methods
- **BERT/DistilBERT** - Procesamiento de lenguaje natural
- **Scikit-learn** - Pipeline de ML

### **Procesamiento de Datos**
- **Pandas** - Manipulación de datos
- **NumPy** - Computación numérica
- **PySpark** - Procesamiento distribuido
- **SQLAlchemy** - ORM para bases de datos

### **Visualización**
- **Dash** - Dashboards interactivos
- **Plotly** - Gráficos interactivos
- **Matplotlib/Seaborn** - Visualizaciones estáticas
- **Power BI/Tableau** - Dashboards empresariales

### **Explicabilidad**
- **SHAP** - Explicaciones de modelos
- **LIME** - Interpretabilidad local
- **Feature Importance** - Importancia de características

## 📈 **Métricas de Rendimiento**

### **Detección de Fraude**
- **Accuracy:** 92.25%
- **ROC AUC:** 0.8782
- **Precision:** 21.61%
- **Recall:** 78.79%

### **Análisis de Twitter**
- **Tiempo de procesamiento:** < 2 segundos por tweet
- **Precisión de sentimientos:** 85%+
- **Escalabilidad:** 1000+ tweets por minuto

### **Sistema de Recomendaciones**
- **Precisión de recomendaciones:** 78%
- **Cobertura de productos:** 95%
- **Diversidad de recomendaciones:** 0.82

## 🤝 **Contribución**

### **Cómo Contribuir**
1. **Fork** el repositorio
2. **Crear** una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. **Commit** tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. **Push** a la rama (`git push origin feature/nueva-funcionalidad`)
5. **Crear** un Pull Request

### **Estándares de Código**
- **Python:** PEP 8
- **Documentación:** Docstrings en inglés
- **Tests:** Cobertura mínima del 80%
- **Commits:** Mensajes descriptivos en inglés

## 📄 **Licencia**

Este proyecto está bajo la licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 📞 **Contacto**

- **Autor:** [Tu Nombre]
- **Email:** [tu.email@ejemplo.com]
- **LinkedIn:** [Tu LinkedIn]
- **GitHub:** [Tu GitHub]

## 🙏 **Agradecimientos**

- **Comunidad de Machine Learning** por las librerías utilizadas
- **Contribuidores** que han ayudado al desarrollo
- **Usuarios** que han proporcionado feedback valioso

---

## 🎯 **Estado del Proyecto**

| Sistema | Estado | Última Actualización | Documentación |
|---------|--------|---------------------|---------------|
| Detección de Fraude | ✅ Funcionando | 25/01/2024 | ✅ Completa |
| Análisis de Twitter | ✅ Funcionando | 25/01/2024 | ✅ Completa |
| Sistema de Recomendaciones | ✅ Funcionando | 25/01/2024 | ✅ Completa |
| Dashboard de Ventas | ✅ Funcionando | 25/01/2024 | ✅ Completa |

**¡SmartRetail está listo para producción!** 🚀

---

*Última actualización: 25/01/2024* 