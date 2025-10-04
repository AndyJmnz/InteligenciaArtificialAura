from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from paddleocr import PaddleOCR
import requests
import tempfile
import os
from dotenv import load_dotenv

# ---------------------------
# CONFIGURACIÓN DE OPENROUTER
# ---------------------------

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = os.getenv("OPENROUTER_API_URL")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

app = FastAPI(title="OCR + NLP API", description="API con PaddleOCR y DeepSeek via OpenRouter")

# Inicializar PaddleOCR con configuración optimizada
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='es',
    use_gpu=False,
    show_log=False,
    # Parámetros optimizados para mejor precisión
    det_db_thresh=0.3,      # Umbral de detección más bajo
    det_db_box_thresh=0.5,  # Umbral de caja más bajo
    det_db_unclip_ratio=1.6, # Mejor expansión de cajas
    rec_batch_num=6         # Procesar más texto en lotes
)

# ---------------------------
# FUNCIONES INTERNAS
# ---------------------------

def preprocess_image_simple(image_path: str):
    """Preprocesa la imagen para mejorar OCR"""
    try:
        # Leer imagen
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Reducir ruido
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Guardar imagen procesada
        processed_path = image_path.replace('.', '_processed.')
        cv2.imwrite(processed_path, denoised)
        
        return processed_path
        
    except Exception as e:
        print(f"Error en preprocesamiento: {e}")
        return image_path

def preprocess_image_advanced(image_path: str):
    """Preprocesamiento avanzado para mejorar OCR"""
    try:
        # Leer imagen
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
        # 1. Redimensionar si es muy pequeña o muy grande
        height, width = img.shape[:2]
        if width < 300 or height < 300:
            scale_factor = max(300/width, 300/height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        elif width > 2000 or height > 2000:
            scale_factor = min(2000/width, 2000/height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # 2. Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Aplicar filtro bilateral para reducir ruido manteniendo bordes
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 4. Mejorar contraste adaptativo
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(bilateral)
        
        # 5. Binarización adaptativa (múltiples métodos)
        # Método 1: Otsu
        _, binary1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Método 2: Adaptativo
        binary2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # 6. Operaciones morfológicas para limpiar
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary1 = cv2.morphologyEx(binary1, cv2.MORPH_CLOSE, kernel)
        binary2 = cv2.morphologyEx(binary2, cv2.MORPH_CLOSE, kernel)
        
        # Guardar múltiples versiones
        processed_paths = []
        base_name = image_path.replace('.', '_processed')
        
        # Versión mejorada original
        enhanced_path = f"{base_name}_enhanced.jpg"
        cv2.imwrite(enhanced_path, enhanced)
        processed_paths.append(enhanced_path)
        
        # Versión binaria Otsu
        otsu_path = f"{base_name}_otsu.jpg"
        cv2.imwrite(otsu_path, binary1)
        processed_paths.append(otsu_path)
        
        # Versión binaria adaptativa
        adaptive_path = f"{base_name}_adaptive.jpg"
        cv2.imwrite(adaptive_path, binary2)
        processed_paths.append(adaptive_path)
        
        return processed_paths
        
    except Exception as e:
        print(f"Error en preprocesamiento avanzado: {e}")
        return [image_path]

def send_to_deepseek(extracted_text: str):
    """Envía el texto extraído a DeepSeek para corrección"""
    try:
        prompt = f"""Corrige y mejora este texto extraído por OCR. Mantén el significado original y responde solo con el texto corregido:

Texto: {extracted_text}

Texto corregido:"""

        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            print(f"Error en API: {response.status_code}")
            return extracted_text
            
    except Exception as e:
        print(f"Error con DeepSeek: {e}")
        return extracted_text

def send_to_deepseek_improved(extracted_text: str):
    """Versión mejorada para envío a DeepSeek"""
    try:
        # Verificar si el texto es muy corto o vacío
        if not extracted_text.strip() or len(extracted_text.strip()) < 3:
            return "No se pudo extraer texto legible de la imagen."
        
        prompt = f"""Eres un experto en corrección de texto extraído por OCR. Tu tarea es corregir ÚNICAMENTE errores de OCR manteniendo el contenido y formato original.

REGLAS:
- Corrige solo errores obvios de OCR (caracteres mal reconocidos)
- Mantén el formato y estructura original
- NO agregues información nueva
- NO cambies el significado
- Si hay números o códigos, mantenlos exactos
- Conserva saltos de línea y espaciado

Texto extraído por OCR:
{extracted_text}

Texto corregido (solo corrección de errores OCR):"""

        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "Eres un corrector de texto OCR experto. Solo corriges errores de reconocimiento de caracteres, no cambias el contenido."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": min(len(extracted_text) * 2, 2000),  # Límite dinámico
            "top_p": 0.9
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
        
        if response.status_code == 200:
            result = response.json()
            corrected = result['choices'][0]['message']['content'].strip()
            
            # Validación adicional
            if len(corrected) > len(extracted_text) * 3:
                # Si la respuesta es demasiado larga, probablemente agregó contenido
                return extracted_text
            
            return corrected
        else:
            print(f"Error en API DeepSeek: {response.status_code}")
            return extracted_text
            
    except Exception as e:
        print(f"Error enviando a DeepSeek: {e}")
        return extracted_text

def extract_text_ultra_simple(image_path: str, min_confidence=0.3):
    """ Extrae texto con PaddleOCR de la imagen original y mejorada """
    improved_path = preprocess_image_simple(image_path)
    all_results = []

    for path, source in [(image_path, "original"), (improved_path, "mejorada")]:
        try:
            # Usar el OCR inicializado globalmente
            results = ocr.ocr(path, cls=True)
            
            if results and results[0]:  # Verificar que hay resultados
                for detection in results[0]:
                    if detection and len(detection) >= 2:
                        bbox, (text, confidence) = detection
                        if confidence >= min_confidence and text.strip():
                            all_results.append({
                                "text": text.strip(),
                                "confidence": float(confidence),
                                "source": source
                            })
        except Exception as e:
            print(f"Error procesando {source}: {e}")

    # Filtrar duplicados conservando mayor confianza
    unique_texts = {}
    for item in all_results:
        text = item["text"]
        if text not in unique_texts or item["confidence"] > unique_texts[text]["confidence"]:
            unique_texts[text] = item

    final_results = list(unique_texts.values())
    complete_text = " ".join([item["text"] for item in final_results])
    
    # Limpiar archivo temporal mejorado
    try:
        if os.path.exists(improved_path) and improved_path != image_path:
            os.remove(improved_path)
    except:
        pass
    
    return complete_text, final_results

def extract_text_improved(image_path: str, min_confidence=0.2):
    """Extracción de texto mejorada con múltiples procesamiento"""
    all_results = []
    processed_paths = preprocess_image_advanced(image_path)
    
    # Procesar imagen original + todas las versiones procesadas
    images_to_process = [
        (image_path, "original"),
        *[(path, f"processed_{i}") for i, path in enumerate(processed_paths)]
    ]
    
    for path, source in images_to_process:
        try:
            # OCR con diferentes configuraciones
            results = ocr.ocr(path, cls=True)
            
            if results and results[0]:
                for detection in results[0]:
                    if detection and len(detection) >= 2:
                        bbox, (text, confidence) = detection
                        if confidence >= min_confidence and text.strip():
                            # Limpiar texto básico
                            clean_text = text.strip()
                            clean_text = ' '.join(clean_text.split())  # Normalizar espacios
                            
                            all_results.append({
                                "text": clean_text,
                                "confidence": float(confidence),
                                "source": source,
                                "bbox": bbox
                            })
        except Exception as e:
            print(f"Error procesando {source}: {e}")
    
    # Mejorar deduplicación con similitud de texto
    unique_texts = {}
    for item in all_results:
        text = item["text"].lower().strip()
        
        # Buscar texto similar existente
        found_similar = False
        for existing_text in unique_texts.keys():
            # Calcular similitud simple
            if calculate_text_similarity(text, existing_text) > 0.8:
                # Mantener el de mayor confianza
                if item["confidence"] > unique_texts[existing_text]["confidence"]:
                    del unique_texts[existing_text]
                    unique_texts[text] = item
                found_similar = True
                break
        
        if not found_similar:
            unique_texts[text] = item
    
    # Limpiar archivos temporales
    for path in processed_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except:
            pass
    
    final_results = list(unique_texts.values())
    # Ordenar por posición Y (arriba a abajo) para mejor orden de lectura
    final_results.sort(key=lambda x: x["bbox"][0][1] if x.get("bbox") else 0)
    
    complete_text = " ".join([item["text"] for item in final_results])
    return complete_text, final_results

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calcula similitud simple entre dos textos"""
    if not text1 or not text2:
        return 0.0
    
    # Convertir a conjuntos de palabras
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calcular intersección
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

# ---------------------------
# ENDPOINTS FASTAPI
# ---------------------------

@app.get("/")
async def root():
    """Endpoint principal para verificar que la API está funcionando"""
    return {
        "message": "OCR + NLP API está funcionando",
        "version": "1.0.0",
        "engine": "PaddleOCR",
        "nlp": "DeepSeek via OpenRouter",
        "endpoints": {
            "ocr": "/ocr/ (POST) - Procesar imagen con OCR",
            "health": "/health (GET) - Estado de salud",
            "docs": "/docs - Documentación automática"
        }
    }

@app.get("/health")
async def health_check():
    """Endpoint de salud para monitoreo"""
    try:
        # Verificar que PaddleOCR funciona
        test_status = "OK" if ocr else "ERROR"
        
        return {
            "status": "healthy",
            "ocr_engine": "PaddleOCR", 
            "ocr_status": test_status,
            "api_version": "1.0.0",
            "timestamp": "2025-10-02"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "ocr_engine": "PaddleOCR"
            }
        )

@app.post("/ocr/")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        # Guardar archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Extraer texto
        extracted_text, details = extract_text_ultra_simple(tmp_path, min_confidence=0.3)

        # Enviar a DeepSeek solo si hay texto
        if extracted_text.strip():
            corrected_text = send_to_deepseek(extracted_text)
        else:
            corrected_text = "No se detectó texto en la imagen."

        # Limpiar archivo temporal
        try:
            os.remove(tmp_path)
        except:
            pass

        return JSONResponse(
            content={
                "texto_extraido": extracted_text,
                "texto_corregido": corrected_text,
                "detalles": details,
                "estadisticas": {
                    "palabras_extraidas": len(extracted_text.split()) if extracted_text.strip() else 0,
                    "detecciones": len(details)
                }
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})