# 📊 Ejemplo Práctico - Dashboard de Ventas y Rentabilidad

## 📋 Información del Proyecto

**Nombre del Dashboard:** Dashboard de Ventas y Rentabilidad - SmartRetail  
**Herramienta Utilizada:** Power BI  
**Fecha de Creación:** 15/01/2024  
**Versión:** 1.0  
**Autor:** Equipo de Analytics  
**Departamento:** Business Intelligence

---

## 🎯 **1. Objetivo del Dashboard**

### **Contexto del Negocio**
SmartRetail necesita un dashboard integral para monitorear el rendimiento de ventas y rentabilidad en tiempo real. El negocio opera en múltiples regiones con diferentes canales de venta (online, retail, mayorista) y requiere visibilidad inmediata sobre el rendimiento de productos, regiones y vendedores.

### **Objetivos Principales**
- **Objetivo 1:** Monitorear ventas diarias y comparar con metas mensuales
- **Objetivo 2:** Identificar productos de alto y bajo rendimiento por región
- **Objetivo 3:** Analizar la rentabilidad por canal de venta y vendedor
- **Objetivo 4:** Detectar tendencias y patrones de venta para optimizar inventario

### **Preguntas de Negocio a Responder**
1. **Pregunta 1:** ¿Qué productos están generando mayor rentabilidad por región?
2. **Pregunta 2:** ¿Cuáles son los vendedores de mejor rendimiento y qué factores contribuyen?
3. **Pregunta 3:** ¿Cómo se comportan las ventas por canal y qué oportunidades de optimización existen?

### **Audiencia Objetivo**
- **Audiencia Primaria:** Directores de Ventas, Gerentes Regionales, Ejecutivos de Negocio
- **Audiencia Secundaria:** Analistas de Negocio, Equipo de Marketing, Finanzas
- **Frecuencia de Uso:** Diario para operaciones, Semanal para análisis estratégico

---

## 📊 **2. Dataset**

### **Fuentes de Datos**
| Fuente | Tipo | Frecuencia de Actualización | Descripción |
|--------|------|----------------------------|-------------|
| ERP_Sales | Base de datos SQL Server | Diario a las 6:00 AM | Transacciones de ventas principales |
| CRM_System | API REST | Tiempo real | Datos de clientes y oportunidades |
| Inventory_System | Base de datos PostgreSQL | Cada 2 horas | Niveles de inventario y costos |
| Marketing_Data | Archivo Excel | Semanal | Datos de campañas y leads |

### **Estructura de Datos**
```sql
-- Tabla principal de ventas
CREATE TABLE sales_transactions (
    transaction_id INT PRIMARY KEY,
    sale_date DATE,
    product_id INT,
    product_name VARCHAR(100),
    category VARCHAR(50),
    quantity INT,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    cost_amount DECIMAL(10,2),
    profit_margin DECIMAL(10,2),
    region VARCHAR(50),
    store_id INT,
    salesperson_id INT,
    customer_id INT,
    channel VARCHAR(20),
    payment_method VARCHAR(30)
);
```

### **Campos Clave**
| Campo | Tipo | Descripción | Uso en el Dashboard |
|-------|------|-------------|-------------------|
| sale_date | DATE | Fecha de venta | Filtros temporales y tendencias |
| total_amount | DECIMAL | Monto total de venta | KPIs de ventas |
| profit_margin | DECIMAL | Margen de ganancia | Análisis de rentabilidad |
| region | VARCHAR | Región de venta | Segmentación geográfica |
| channel | VARCHAR | Canal de venta | Análisis por canal |

### **Transformaciones de Datos**
- **Limpieza:** Eliminación de transacciones duplicadas y valores nulos
- **Agregaciones:** Cálculo de ventas diarias, semanales y mensuales
- **Cálculos:** Margen de ganancia, tasa de conversión, promedio de ticket
- **Filtros:** Exclusión de transacciones de prueba y devoluciones

---

## 📈 **3. KPIs Seleccionados**

### **KPIs Principales**

#### **KPI 1: Ventas Totales Diarias**
- **Fórmula:** `SUM(total_amount) WHERE sale_date = TODAY`
- **Objetivo:** $50,000 diarios
- **Actual:** $47,850
- **Tendencia:** Mejorando (+5% vs semana anterior)
- **Importancia:** Alta

#### **KPI 2: Margen de Ganancia Promedio**
- **Fórmula:** `AVERAGE(profit_margin)`
- **Objetivo:** 25%
- **Actual:** 23.5%
- **Tendencia:** Estable
- **Importancia:** Alta

#### **KPI 3: Tasa de Conversión por Canal**
- **Fórmula:** `(Ventas / Leads) * 100`
- **Objetivo:** 15%
- **Actual:** 12.8%
- **Tendencia:** Empeorando (-2% vs mes anterior)
- **Importancia:** Media

### **KPIs Secundarios**
| KPI | Fórmula | Objetivo | Actual | Estado |
|-----|---------|----------|--------|--------|
| Ticket Promedio | `AVERAGE(total_amount)` | $150 | $142 | ⚠️ |
| Productos Vendidos | `COUNT(DISTINCT product_id)` | 500 | 485 | ✅ |
| Clientes Únicos | `COUNT(DISTINCT customer_id)` | 200 | 195 | ✅ |

---

## 📊 **4. Visualizaciones Implementadas**

### **Página 1: Resumen Ejecutivo**

#### **Visualización 1: KPIs Principales**
- **Título:** "Métricas Clave del Día"
- **Tipo:** Tarjetas de métricas con indicadores de tendencia
- **Ejes:** 
  - **Eje X:** N/A (tarjetas)
  - **Eje Y:** Valores de KPIs
- **Filtros:** Fecha, Región
- **Interactividad:** Drill-down a detalles por región
- **Propósito:** Vista rápida del rendimiento general

#### **Visualización 2: Ventas por Región**
- **Título:** "Ventas Totales por Región"
- **Tipo:** Gráfico de barras horizontales
- **Ejes:** 
  - **Eje X:** Monto de ventas
  - **Eje Y:** Regiones
- **Filtros:** Período de tiempo, Canal
- **Interactividad:** Hover para detalles, click para drill-down
- **Propósito:** Identificar regiones de mejor y peor rendimiento

### **Página 2: Análisis de Productos**

#### **Visualización 3: Top Productos por Rentabilidad**
- **Título:** "Productos con Mayor Margen de Ganancia"
- **Tipo:** Gráfico de barras con línea de tendencia
- **Ejes:** 
  - **Eje X:** Productos
  - **Eje Y:** Margen de ganancia (%)
- **Filtros:** Categoría, Región, Período
- **Interactividad:** Filtro cruzado con otras visualizaciones
- **Propósito:** Identificar productos más rentables para optimizar inventario

### **Elementos de Navegación**
- **Filtros Globales:** Fecha (últimos 7 días, 30 días, 90 días), Región, Canal
- **Botones de Navegación:** Resumen, Productos, Vendedores, Rentabilidad
- **Drill-down:** De región a tienda, de producto a variante
- **Cross-filtering:** Selección de región filtra todas las visualizaciones

---

## 📸 **5. Capturas con Explicación**

### **Captura 1: Vista General del Dashboard**
![Dashboard General](images/dashboard_overview.png)

**Explicación:**
- **Elementos mostrados:** KPIs principales, gráfico de ventas por región, tendencia temporal
- **KPIs destacados:** Ventas diarias ($47,850), Margen (23.5%), Conversión (12.8%)
- **Interactividad:** Click en región para ver detalles, hover para tooltips
- **Insights clave:** Región Norte lidera ventas, Sur tiene menor margen

### **Captura 2: Detalle de KPI Principal**
![KPI Ventas](images/sales_kpi.png)

**Explicación:**
- **Métrica mostrada:** Ventas totales del día con comparación vs objetivo
- **Tendencia:** +5% vs semana anterior, -4.3% vs objetivo
- **Comparación:** Gráfico de tendencia de últimos 30 días
- **Acciones sugeridas:** Revisar estrategia en regiones con bajo rendimiento

### **Captura 3: Análisis Detallado**
![Análisis Productos](images/product_analysis.png)

**Explicación:**
- **Gráfico mostrado:** Top 10 productos por margen de ganancia
- **Patrones identificados:** Productos premium tienen mayor margen
- **Outliers:** Producto "Laptop Pro" con 35% de margen
- **Correlaciones:** Productos con mayor precio tienen mejor margen

### **Captura 4: Filtros y Navegación**
![Filtros](images/filters.png)

**Explicación:**
- **Filtros disponibles:** Fecha, Región, Canal, Categoría
- **Funcionalidad:** Filtros afectan todas las visualizaciones simultáneamente
- **Impacto:** Permite análisis granular por segmentos
- **Mejores prácticas:** Usar filtros para enfocar análisis en áreas específicas

---

## 🔧 **6. Configuración Técnica**

### **Herramienta Utilizada**
- **Software:** Power BI Desktop
- **Versión:** 2.123.456.0
- **Conectores:** SQL Server, REST API, Excel
- **Servidor:** Power BI Service (Premium)

### **Configuración de Datos**
```yaml
Data Sources:
  - Type: SQL Server
    Connection: Server=BI-SERVER;Database=SalesDB;Trusted_Connection=true;
    Refresh Schedule: Daily at 6:00 AM
  - Type: REST API
    URL: https://api.smartretail.com/sales
    Authentication: OAuth2
    Refresh Schedule: Every 2 hours
  - Type: Excel File
    Location: \\server\data\marketing_data.xlsx
    Refresh Schedule: Weekly on Monday
```

### **Medidas y Cálculos**
| Medida | Fórmula | Descripción |
|--------|---------|-------------|
| Ventas Totales | `SUM(sales[total_amount])` | Suma de todas las ventas |
| Margen Promedio | `AVERAGE(sales[profit_margin])` | Margen promedio de ganancia |
| Tasa de Conversión | `DIVIDE([Ventas], [Leads], 0)` | Ratio de conversión |
| Crecimiento YoY | `DIVIDE([Ventas Actual] - [Ventas Anterior], [Ventas Anterior])` | Crecimiento interanual |

### **Optimizaciones de Rendimiento**
- **Filtros aplicados:** Filtro de fecha para últimos 2 años
- **Agregaciones:** Tablas calculadas para métricas frecuentes
- **Índices:** Índices en fecha y región en base de datos
- **Caché:** Configuración de caché de 1 hora

---

## 📊 **7. Métricas de Uso**

### **Adopción del Dashboard**
- **Usuarios únicos:** 45 usuarios activos
- **Sesiones por día:** 12 sesiones promedio
- **Tiempo promedio de sesión:** 8.5 minutos
- **Páginas más visitadas:** Resumen Ejecutivo (60%), Análisis de Productos (25%)

### **Feedback de Usuarios**
| Aspecto | Calificación (1-5) | Comentarios |
|---------|-------------------|-------------|
| Facilidad de uso | 4.2/5 | "Interfaz intuitiva, fácil navegación" |
| Relevancia de datos | 4.5/5 | "KPIs muy útiles para toma de decisiones" |
| Velocidad de carga | 3.8/5 | "Algunas visualizaciones tardan en cargar" |
| Calidad de insights | 4.3/5 | "Proporciona insights valiosos rápidamente" |

---

## 🎯 **8. Reflexión Final**

### **Logros Alcanzados**
- ✅ **Objetivo 1:** Dashboard operativo con actualización diaria de ventas
- ✅ **Objetivo 2:** Identificación de top 10 productos por rentabilidad
- ✅ **Objetivo 3:** Análisis detallado de rendimiento por canal y vendedor
- ✅ **Objetivo 4:** Detección de tendencias de venta para optimización

### **Desafíos Enfrentados**
- **Desafío 1:** Integración de múltiples fuentes de datos - Resuelto con ETL optimizado
- **Desafío 2:** Rendimiento con grandes volúmenes de datos - Resuelto con agregaciones y caché
- **Desafío 3:** Adopción por parte de usuarios - Resuelto con capacitación y documentación

### **Lecciones Aprendidas**
- **Lección 1:** La simplicidad en el diseño aumenta la adopción del dashboard
- **Lección 2:** Los filtros globales mejoran significativamente la experiencia de usuario
- **Lección 3:** La actualización en tiempo real es crítica para la credibilidad

### **Impacto en el Negocio**
- **Decisión 1:** Reasignación de inventario basada en análisis de productos
- **Ahorro/Ingreso:** $150,000 en optimización de inventario
- **Eficiencia:** 30% reducción en tiempo de análisis de ventas
- **Visibilidad:** Mejor comprensión de patrones de venta por región

### **Mejoras Futuras**
- **Mejora 1:** Integración con sistema de predicción de demanda
- **Mejora 2:** Alertas automáticas para KPIs fuera de rango
- **Mejora 3:** Dashboard móvil para acceso en campo

### **Recomendaciones**
- **Para usuarios:** Revisar dashboard diariamente, usar filtros para análisis específicos
- **Para desarrolladores:** Implementar alertas automáticas, optimizar consultas
- **Para stakeholders:** Usar insights para decisiones estratégicas de expansión

---

## 📋 **9. Anexos**

### **Anexo A: Glosario de Términos**
| Término | Definición |
|---------|------------|
| Margen de Ganancia | (Precio de Venta - Costo) / Precio de Venta |
| Ticket Promedio | Ventas Totales / Número de Transacciones |
| Tasa de Conversión | Ventas / Leads * 100 |
| YoY | Year over Year (comparación interanual) |

### **Anexo B: Códigos de Color**
| Color | Significado | Uso |
|-------|-------------|-----|
| Verde | Objetivo alcanzado | KPIs en rango |
| Rojo | Objetivo no alcanzado | KPIs fuera de rango |
| Amarillo | Advertencia | KPIs cerca del límite |

### **Anexo C: Contactos**
| Rol | Nombre | Email | Teléfono |
|-----|--------|-------|----------|
| Product Owner | María González | mgonzalez@smartretail.com | +1-555-0101 |
| Desarrollador BI | Carlos Rodríguez | crodriguez@smartretail.com | +1-555-0102 |

---

## 📝 **10. Historial de Cambios**

| Versión | Fecha | Autor | Cambios |
|---------|-------|-------|---------|
| 1.0 | 15/01/2024 | Equipo Analytics | Versión inicial |
| 1.1 | 20/01/2024 | Carlos Rodríguez | Agregado filtro de canal |
| 1.2 | 25/01/2024 | María González | Optimización de rendimiento |

---

## 🎯 **Conclusión**

Este dashboard ha sido diseñado para proporcionar visibilidad completa del rendimiento de ventas y rentabilidad de SmartRetail y ha demostrado ser una herramienta valiosa para la toma de decisiones operativas y estratégicas. A través de KPIs claros, visualizaciones intuitivas y análisis detallados, hemos logrado mejorar la eficiencia operativa y la rentabilidad del negocio que impacta positivamente en todas las áreas de la empresa.

**El dashboard está listo para uso en producción y se recomienda su adopción por parte de todos los stakeholders del negocio.**

---

*Documento creado el 15/01/2024 - Última actualización: 25/01/2024* 