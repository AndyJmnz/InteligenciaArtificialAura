# 🐳 Docker - Guía de Uso para AuraAI

## ✅ Estado Actual
Tu aplicación está **funcionando correctamente** en Docker.

## 🚀 Comandos Principales

### Construcción y Ejecución
```bash
# Construir imagen
docker build -t aura-ai .

# Ejecutar contenedor
docker run -d -p 8000:8000 --name aura-api aura-ai

# O usar el script automático
build-docker.bat
```

### Gestión del Contenedor
```bash
# Ver contenedores activos
docker ps

# Ver logs
docker logs aura-api
docker logs -f aura-api  # En tiempo real

# Detener
docker stop aura-api

# Reiniciar
docker restart aura-api

# Eliminar
docker rm aura-api
```

### Reconstruir después de cambios
```bash
docker stop aura-api
docker rm aura-api
docker build -t aura-ai . --no-cache
docker run -d -p 8000:8000 --name aura-api aura-ai
```

## 🧪 Probar la API

### Health Check
```bash
curl http://localhost:8000/health
# Respuesta: {"status":"ok"}
```

### Documentación Automática
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoint OCR
```bash
curl -X POST "http://localhost:8000/ocr/" \
     -F "file=@imagen.jpg"
```

## 📁 Archivos Creados/Modificados

- ✅ `Dockerfile` - Configuración optimizada para funcionar
- ✅ `requirements.txt` - Dependencias compatibles
- ✅ `build-docker.bat` - Script automático de construcción
- ✅ `.gitignore` - Archivos a ignorar

## 🔧 Configuración Final

### Dockerfile
- Base: `python:3.11-slim`
- Dependencias mínimas del sistema
- Puerto: 8000
- Health check incluido

### Requirements.txt
- Versiones exactas y compatibles
- OpenCV headless (sin interfaz gráfica)
- PaddleOCR funcionando

## ⚠️ Notas Importantes

1. **Puerto**: La API corre en puerto 8000
2. **Salud**: El health check está en `/health`
3. **Logs**: Usa `docker logs aura-api` para debug
4. **Performance**: Primera ejecución es más lenta (descarga modelos OCR)

## 🎉 ¡Todo Funcionando!

Tu API OCR está lista para:
- ✅ Desarrollo local con Docker
- ✅ Despliegue en Railway
- ✅ Procesamiento de imágenes
- ✅ Corrección de texto con IA