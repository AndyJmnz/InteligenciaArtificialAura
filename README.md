# AuraAI - OCR + NLP API

API que combina PaddleOCR para extracción de texto de imágenes y DeepSeek (vía OpenRouter) para corrección ortográfica y gramatical.

## Características

- **OCR**: Extracción de texto de imágenes usando PaddleOCR
- **Procesamiento de imágenes**: Mejora automática de contraste y nitidez
- **NLP**: Corrección de texto usando DeepSeek vía OpenRouter
- **API REST**: Interfaz FastAPI para fácil integración

## Instalación Local

### Usando Docker (Recomendado)

```bash
# Construir la imagen
docker build -t auraai .

# Ejecutar el contenedor
docker run -p 8000:8000 auraai
```

### Instalación Manual

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Despliegue en Railway

### Opción 1: Desde GitHub (Recomendado)

1. Haz push de tu código a GitHub
2. Ve a [Railway](https://railway.app)
3. Conecta tu repositorio de GitHub
4. Railway detectará automáticamente el `Dockerfile` y desplegará la aplicación

### Opción 2: Railway CLI

```bash
# Instalar Railway CLI
npm install -g @railway/cli

# Login en Railway
railway login

# Inicializar proyecto
railway init

# Desplegar
railway up
```

### Variables de Entorno

Asegúrate de configurar las siguientes variables de entorno en Railway:

- `API_KEY`: Tu clave API de OpenRouter (actualmente hardcodeada en el código)

## Uso de la API

### Endpoint Principal

**POST** `/ocr/`

Sube una imagen y obtén el texto extraído y corregido.

```bash
curl -X POST "https://tu-app.railway.app/ocr/" \
     -F "file=@imagen.jpg"
```

**Respuesta:**
```json
{
  "texto_extraido": "Texto original extraído de la imagen",
  "texto_corregido": "Texto corregido por DeepSeek",
  "detalles": [
    {
      "text": "palabra",
      "confidence": 0.95,
      "source": "original"
    }
  ]
}
```

### Endpoints de Salud

- **GET** `/` - Health check básico
- **GET** `/health` - Status de la aplicación

## Arquitectura

```
Usuario → FastAPI → PaddleOCR → DeepSeek (OpenRouter) → Respuesta
```

1. **Recepción**: FastAPI recibe la imagen
2. **Preprocesamiento**: Mejora de imagen (contraste, nitidez)
3. **OCR**: PaddleOCR extrae texto de imagen original y mejorada
4. **Filtrado**: Eliminación de duplicados por confianza
5. **NLP**: DeepSeek corrige ortografía y gramática
6. **Respuesta**: JSON con texto original y corregido

## Tecnologías

- **FastAPI**: Framework web Python
- **PaddleOCR**: Reconocimiento óptico de caracteres
- **OpenCV**: Procesamiento de imágenes
- **PIL/Pillow**: Manipulación de imágenes
- **OpenRouter**: Acceso a modelos de IA (DeepSeek)
- **Docker**: Containerización
- **Railway**: Plataforma de despliegue

## Configuración Recomendada para Producción

### Seguridad
- [ ] Mover API_KEY a variables de entorno
- [ ] Implementar autenticación
- [ ] Añadir rate limiting
- [ ] Validar tipos de archivo

### Performance
- [ ] Implementar cache para resultados
- [ ] Optimizar procesamiento de imágenes
- [ ] Configurar timeout para requests

### Monitoreo
- [ ] Añadir logs estructurados
- [ ] Métricas de performance
- [ ] Health checks más robustos

## Licencia

MIT License