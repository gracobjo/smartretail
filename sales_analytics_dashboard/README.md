# Dashboard de Análisis de Ventas y Clientes - SmartRetail

## Descripción del Proyecto

Este dashboard interactivo implementa un sistema completo de análisis de ventas y clientes con KPIs avanzados:

1. **Customer Lifetime Value (CLV)**: Valor del cliente a lo largo del tiempo
2. **Churn Rate**: Tasa de abandono de clientes
3. **Análisis de Cohortes**: Segmentación temporal de clientes
4. **KPIs de Ventas**: Métricas de rendimiento comercial
5. **Segmentación de Clientes**: Análisis por segmentos

## Características Principales

### KPIs Avanzados Implementados

#### 1. Customer Lifetime Value (CLV)
- **CLV Histórico**: Valor total generado por cliente
- **CLV Predictivo**: Predicción de valor futuro
- **CLV por Segmento**: Análisis por grupos de clientes
- **CLV Evolution**: Evolución temporal del CLV

#### 2. Churn Analysis
- **Churn Rate**: Tasa de abandono mensual/trimestral
- **Churn Prediction**: Predicción de clientes en riesgo
- **Churn Reasons**: Análisis de causas de abandono
- **Retention Strategies**: Estrategias de retención

#### 3. Cohort Analysis
- **Cohort Retention**: Retención por cohortes
- **Cohort Revenue**: Ingresos por cohortes
- **Cohort Size**: Tamaño de cohortes
- **Cohort Behavior**: Comportamiento por cohortes

#### 4. Sales KPIs
- **Revenue Metrics**: Métricas de ingresos
- **Conversion Rates**: Tasas de conversión
- **Average Order Value**: Valor promedio de pedido
- **Sales Growth**: Crecimiento de ventas

### Tecnologías Utilizadas

#### Frontend y Visualización
- **Dash**: Framework web para dashboards
- **Plotly**: Gráficos interactivos
- **Bootstrap**: Diseño responsivo
- **CSS/HTML**: Personalización de estilos

#### Backend y Procesamiento
- **Python**: Lógica de negocio
- **Pandas**: Manipulación de datos
- **NumPy**: Cálculos numéricos
- **Scikit-learn**: Análisis predictivo

#### Almacenamiento y Datos
- **SQLite**: Base de datos local
- **CSV/JSON**: Archivos de datos
- **Pandas**: Procesamiento de datos
- **NumPy**: Operaciones matemáticas

## Estructura del Proyecto

```
sales_analytics_dashboard/
├── src/
│   ├── data/
│   │   ├── data_generator.py       # Generador de datos simulados
│   │   ├── data_processor.py       # Procesamiento de datos
│   │   └── database.py             # Conexiones a BD
│   ├── analytics/
│   │   ├── clv_calculator.py       # Cálculo de CLV
│   │   ├── churn_analyzer.py       # Análisis de churn
│   │   ├── cohort_analyzer.py      # Análisis de cohortes
│   │   └── kpi_calculator.py       # Cálculo de KPIs
│   ├── visualization/
│   │   ├── dashboard.py            # Dashboard principal
│   │   ├── charts.py               # Componentes de gráficos
│   │   └── filters.py              # Filtros interactivos
│   └── utils/
│       ├── config.py               # Configuración
│       ├── helpers.py              # Utilidades
│       └── constants.py            # Constantes
├── data/
│   ├── raw/                        # Datos crudos
│   ├── processed/                  # Datos procesados
│   └── sample/                     # Datos de ejemplo
├── assets/
│   ├── css/                        # Estilos CSS
│   ├── images/                     # Imágenes
│   └── js/                         # JavaScript
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── kpi_analysis.ipynb
│   └── cohort_analysis.ipynb
├── results/
│   ├── visualizations/             # Visualizaciones
│   ├── reports/                    # Reportes
│   └── exports/                    # Exportaciones
├── tests/
│   ├── test_analytics.py
│   ├── test_data.py
│   └── test_visualization.py
├── config/
│   ├── dashboard_config.yaml       # Configuración del dashboard
│   ├── kpi_config.yaml            # Configuración de KPIs
│   └── data_config.yaml           # Configuración de datos
├── main.py                         # Script principal
├── run_dashboard.py                # Ejecutar dashboard
├── requirements.txt                # Dependencias
└── README.md                       # Documentación
```

## Instalación

### Prerrequisitos

- Python 3.8+
- Dash 2.0+
- Plotly 5.0+
- Pandas 1.3+
- NumPy 1.21+

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone <repository-url>
cd sales_analytics_dashboard

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### 1. Ejecutar Dashboard

```bash
python run_dashboard.py
```

### 2. Generar Datos Simulados

```python
from src.data.data_generator import DataGenerator

generator = DataGenerator()
data = generator.generate_sales_data(n_customers=10000, months=24)
```

### 3. Análisis de KPIs

```python
from src.analytics.kpi_calculator import KPICalculator

calculator = KPICalculator()
clv = calculator.calculate_clv(customer_data)
churn_rate = calculator.calculate_churn_rate(customer_data)
```

## KPIs Implementados

### 1. Customer Lifetime Value (CLV)

#### Definición
El CLV representa el valor total que un cliente generará durante toda su relación con la empresa.

#### Métricas Calculadas
- **CLV Histórico**: Suma de todas las compras realizadas
- **CLV Predictivo**: Predicción basada en comportamiento histórico
- **CLV por Segmento**: Análisis por grupos de clientes
- **CLV Evolution**: Tendencias temporales

#### Fórmulas Utilizadas
```
CLV Histórico = Σ(Valor de Compra × Frecuencia)
CLV Predictivo = CLV Histórico × Factor de Retención
```

### 2. Churn Rate

#### Definición
Porcentaje de clientes que dejan de comprar en un período específico.

#### Métricas Calculadas
- **Churn Rate Mensual**: Abandono mensual
- **Churn Rate Trimestral**: Abandono trimestral
- **Churn Prediction**: Predicción de clientes en riesgo
- **Churn Reasons**: Análisis de causas

#### Fórmulas Utilizadas
```
Churn Rate = (Clientes Perdidos / Total Clientes) × 100
Churn Prediction = Probabilidad de Abandono × 100
```

### 3. Cohort Analysis

#### Definición
Análisis de grupos de clientes que comparten características temporales.

#### Métricas Calculadas
- **Cohort Retention**: Retención por cohortes
- **Cohort Revenue**: Ingresos por cohortes
- **Cohort Size**: Tamaño de cohortes
- **Cohort Behavior**: Comportamiento por cohortes

#### Dimensiones Analizadas
- **Cohort por Mes**: Clientes adquiridos por mes
- **Cohort por Canal**: Clientes por canal de adquisición
- **Cohort por Producto**: Clientes por producto inicial

### 4. Sales KPIs

#### Revenue Metrics
- **Total Revenue**: Ingresos totales
- **Monthly Recurring Revenue (MRR)**: Ingresos recurrentes mensuales
- **Annual Recurring Revenue (ARR)**: Ingresos recurrentes anuales
- **Revenue Growth**: Crecimiento de ingresos

#### Conversion Metrics
- **Conversion Rate**: Tasa de conversión
- **Average Order Value (AOV)**: Valor promedio de pedido
- **Purchase Frequency**: Frecuencia de compra
- **Customer Acquisition Cost (CAC)**: Costo de adquisición

## Visualizaciones Implementadas

### 1. Gráficos de KPIs
- **Gauge Charts**: Indicadores de rendimiento
- **Line Charts**: Tendencias temporales
- **Bar Charts**: Comparaciones
- **Pie Charts**: Distribuciones

### 2. Análisis de Cohortes
- **Cohort Heatmap**: Mapa de calor de retención
- **Cohort Line Chart**: Evolución de cohortes
- **Cohort Bar Chart**: Comparación de cohortes

### 3. Análisis de Clientes
- **Customer Segmentation**: Segmentación de clientes
- **CLV Distribution**: Distribución de CLV
- **Churn Analysis**: Análisis de abandono

### 4. Análisis de Ventas
- **Sales Funnel**: Embudo de ventas
- **Revenue Trends**: Tendencias de ingresos
- **Product Performance**: Rendimiento de productos

## Filtros Interactivos

### 1. Filtros Temporales
- **Date Range**: Rango de fechas
- **Period Selection**: Selección de períodos
- **Time Granularity**: Granularidad temporal

### 2. Filtros de Segmentación
- **Customer Segment**: Segmento de cliente
- **Product Category**: Categoría de producto
- **Sales Channel**: Canal de ventas
- **Geographic Region**: Región geográfica

### 3. Filtros de Métricas
- **KPI Selection**: Selección de KPIs
- **Threshold Filters**: Filtros por umbral
- **Comparison Mode**: Modo de comparación

## Funcionalidades Avanzadas

### 1. Análisis Predictivo
- **CLV Prediction**: Predicción de CLV
- **Churn Prediction**: Predicción de churn
- **Revenue Forecasting**: Pronóstico de ingresos

### 2. Alertas y Notificaciones
- **KPI Alerts**: Alertas de KPIs
- **Threshold Notifications**: Notificaciones por umbral
- **Trend Alerts**: Alertas de tendencias

### 3. Exportación y Reportes
- **PDF Reports**: Reportes en PDF
- **Excel Export**: Exportación a Excel
- **CSV Export**: Exportación a CSV
- **Scheduled Reports**: Reportes programados

## Configuración

### Dashboard Configuration

```yaml
# config/dashboard_config.yaml
dashboard:
  title: "Sales Analytics Dashboard"
  theme: "light"
  refresh_interval: 300  # 5 minutes
  
  layout:
    sidebar_width: 250
    main_content_width: 1200
    chart_height: 400
    
  colors:
    primary: "#1f77b4"
    secondary: "#ff7f0e"
    success: "#2ca02c"
    warning: "#d62728"
    info: "#9467bd"
```

### KPI Configuration

```yaml
# config/kpi_config.yaml
kpis:
  clv:
    calculation_method: "historical"
    prediction_horizon: 12  # months
    discount_rate: 0.1
    
  churn:
    definition: "no_purchase_90_days"
    prediction_threshold: 0.7
    alert_threshold: 0.05
    
  cohorts:
    time_granularity: "month"
    retention_periods: 12
    revenue_analysis: true
```

## API Endpoints

### REST API

```python
# Obtener KPIs
GET /api/kpis?metric=clv&period=monthly

# Obtener análisis de cohortes
GET /api/cohorts?start_date=2023-01-01&end_date=2023-12-31

# Obtener predicciones
GET /api/predictions?metric=churn&horizon=3

# Obtener segmentos
GET /api/segments?criteria=rfm
```

### WebSocket API

```python
# Streaming de KPIs en tiempo real
ws://localhost:8050/ws/kpis

# Streaming de alertas
ws://localhost:8050/ws/alerts
```

## Monitoreo y Logging

### Métricas de Rendimiento

- **Dashboard Load Time**: Tiempo de carga
- **Data Processing Time**: Tiempo de procesamiento
- **User Interactions**: Interacciones de usuario
- **Error Rate**: Tasa de errores

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dashboard.log'),
        logging.StreamHandler()
    ]
)
```

## Escalabilidad

### Performance Optimization

- **Data Caching**: Cache de datos
- **Lazy Loading**: Carga diferida
- **Query Optimization**: Optimización de consultas
- **CDN Integration**: Integración con CDN

### Horizontal Scaling

- **Load Balancing**: Balanceo de carga
- **Database Sharding**: Particionamiento de BD
- **Microservices**: Arquitectura de microservicios
- **Containerization**: Contenedores Docker

## Seguridad

### Data Protection

- **Encryption**: Encriptación de datos
- **Access Control**: Control de acceso
- **Audit Logging**: Logs de auditoría
- **GDPR Compliance**: Cumplimiento GDPR

### Dashboard Security

- **Authentication**: Autenticación de usuarios
- **Authorization**: Autorización por roles
- **Input Validation**: Validación de entrada
- **XSS Protection**: Protección XSS

## Contribución

1. Fork el repositorio
2. Crear una rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit los cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Contacto

- **Email**: contacto@smartretail.com
- **GitHub**: https://github.com/smartretail/sales-analytics-dashboard
- **Documentación**: https://docs.smartretail.com/sales-dashboard 