# 🐳 SmartRetail - Docker Deployment

## Descripción
SmartRetail es una plataforma integral de análisis de datos para retail que incluye:
- Detección de fraude financiero
- Sistema de recomendaciones
- Dashboard de análisis de ventas
- Análisis de sentimientos en redes sociales
- Sistema multimodal (facial + texto)

## 🚀 Despliegue con Docker

### Prerrequisitos
- Docker
- Docker Compose
- Git

### 1. Clonar el repositorio
```bash
git clone https://github.com/gracobjo/smartretail.git
cd smartretail
```

### 2. Construir y ejecutar con Docker Compose
```bash
# Construir todas las imágenes
docker-compose build

# Ejecutar todos los servicios
docker-compose up -d

# Ver logs
docker-compose logs -f
```

### 3. Acceder a los servicios

| Servicio | URL | Descripción |
|----------|-----|-------------|
| Dashboard Principal | http://localhost:8501 | Dashboard de ventas con Streamlit |
| Detección de Fraude | http://localhost:8502 | Sistema de detección de fraude |
| Recomendaciones | http://localhost:8503 | Sistema de recomendaciones |
| Análisis de Sentimientos | http://localhost:8504 | Análisis de Twitter |

### 4. Comandos útiles

```bash
# Detener todos los servicios
docker-compose down

# Reiniciar un servicio específico
docker-compose restart smartretail

# Ver logs de un servicio específico
docker-compose logs fraud-detection

# Ejecutar comandos dentro del contenedor
docker-compose exec smartretail python fraud_detection/run_demo.py

# Limpiar volúmenes y contenedores
docker-compose down -v
docker system prune -a
```

## 🏗️ Arquitectura de Servicios

### Servicios Principales
- **smartretail**: Dashboard principal con Streamlit
- **fraud-detection**: Sistema de detección de fraude
- **recommendation-system**: Sistema de recomendaciones
- **sentiment-analysis**: Análisis de sentimientos

### Servicios de Soporte
- **postgres**: Base de datos PostgreSQL
- **redis**: Caché en memoria

## 📊 Monitoreo y Logs

### Ver logs en tiempo real
```bash
# Todos los servicios
docker-compose logs -f

# Servicio específico
docker-compose logs -f smartretail
```

### Métricas del sistema
```bash
# Uso de recursos
docker stats

# Información de contenedores
docker-compose ps
```

## 🔧 Configuración

### Variables de entorno
Crear archivo `.env`:
```env
ENVIRONMENT=production
PYTHONPATH=/app
POSTGRES_DB=smartretail
POSTGRES_USER=smartretail_user
POSTGRES_PASSWORD=smartretail_pass
```

### Volúmenes
Los datos se persisten en:
- `./data`: Datos del proyecto
- `./results`: Resultados de análisis
- `./logs`: Logs de aplicación

## 🚨 Troubleshooting

### Problemas comunes

1. **Puerto ya en uso**
```bash
# Cambiar puertos en docker-compose.yml
ports:
  - "8505:8501"  # Cambiar 8501 por 8505
```

2. **Error de permisos**
```bash
# En Linux/Mac
sudo chown -R $USER:$USER ./data ./results ./logs
```

3. **Memoria insuficiente**
```bash
# Aumentar memoria en docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
```

4. **Error de dependencias**
```bash
# Reconstruir imagen
docker-compose build --no-cache
```

## 🔒 Seguridad

### Buenas prácticas
- Cambiar contraseñas por defecto
- Usar secrets para credenciales
- Configurar firewall
- Actualizar imágenes regularmente

### Variables de entorno seguras
```bash
# Crear archivo .env.secret
echo "POSTGRES_PASSWORD=tu_password_seguro" > .env.secret
```

## 📈 Escalabilidad

### Escalar servicios
```bash
# Escalar dashboard a 3 instancias
docker-compose up -d --scale smartretail=3
```

### Load balancer
```bash
# Agregar nginx como load balancer
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 🧪 Desarrollo

### Modo desarrollo
```bash
# Usar docker-compose.dev.yml
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Hot reload
```bash
# Montar código fuente para desarrollo
volumes:
  - .:/app
  - /app/venv  # Excluir venv del mount
```

## 📚 Documentación Adicional

- [Guía de desarrollo](docs/development_guide.md)
- [Documentación técnica](docs/technical_documentation.md)
- [API Reference](docs/api_reference.md)

## 🤝 Contribución

1. Fork el proyecto
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📞 Soporte

- Issues: [GitHub Issues](https://github.com/gracobjo/smartretail/issues)
- Email: soporte@smartretail.com
- Documentación: [Wiki](https://github.com/gracobjo/smartretail/wiki) 