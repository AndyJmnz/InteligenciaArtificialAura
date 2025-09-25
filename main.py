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

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = os.getenv("OPENROUTER_API_URL")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

app = FastAPI(title="OCR + NLP API", description="API con PaddleOCR y DeepSeek via OpenRouter")

# ---------------------------
# FUNCIONES INTERNAS
# ---------------------------
def send_to_deepseek(complete_text: str) -> str:
    """ Envía texto a DeepSeek vía OpenRouter """
    prompt = (
        f"Corrige los errores ortográficos y gramaticales del siguiente texto "
        f"y elimina lo duplicado devolviendo únicamente el texto corregido, no quiero nada mas que el texto corregido:\n\n{complete_text}"
    )

    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(API_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error en OpenRouter: {response.status_code} - {response.text}")


def preprocess_image_simple(image_path: str) -> str:
    """ Mejora contraste y nitidez de una imagen """
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"No se pudo cargar la imagen {image_path}")

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(1.3)
    img_pil = img_pil.filter(ImageFilter.SHARPEN)

    improved_path = image_path.replace(".jpg", "_mejorada.jpg")
    img_pil.save(improved_path, "JPEG", quality=95)
    return improved_path


def extract_text_ultra_simple(image_path: str, min_confidence=0.3):
    """ Extrae texto con PaddleOCR de la imagen original y mejorada """
    improved_path = preprocess_image_simple(image_path)
    ocr = PaddleOCR(use_angle_cls=True, lang='es')
    all_results = []

    for path, source in [(image_path, "original"), (improved_path, "mejorada")]:
        try:
            # PaddleOCR.ocr() devuelve una lista de resultados
            result = ocr.ocr(path, cls=False)
            if result and len(result) > 0:
                # result[0] contiene las detecciones de texto
                for line in result[0]:
                    if line:
                        # Cada línea tiene: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
                        bbox, (text, confidence) = line
                        if confidence >= min_confidence:
                            all_results.append(
                                {"text": text, "confidence": float(confidence), "source": source}
                            )
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
    return complete_text, final_results


# ---------------------------
# ENDPOINTS FASTAPI
# ---------------------------
@app.post("/ocr/")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        # Guardar archivo temporalmente
        print("File received:", file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Extraer texto
        extracted_text, details = extract_text_ultra_simple(tmp_path, min_confidence=0.3)

        # Enviar a DeepSeek
        corrected_text = send_to_deepseek(extracted_text)

        return JSONResponse(
            content={
                "texto_extraido": extracted_text,
                "texto_corregido": corrected_text,
                "detalles": details,
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------
# HEALTH CHECK ENDPOINT
# ---------------------------
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "OCR + NLP API is running"}


@app.get("/health")
async def health():
    return {"status": "ok"}
