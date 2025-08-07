
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import io
import uvicorn
from pydantic import BaseModel
from typing import Dict
import cv2

app = FastAPI(title="COVID-19 Tanı API'si", description="Akciğer görüntülerinden COVID-19 tanısı")


model = keras.models.load_model('models/covid_cnn_model.h5')


CLASS_NAMES = {
    0: "Normal",
    1: "COVID-19", 
    2: "Pneumonia"
}

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    EĞİTİM İLE TAM UYUMLU preprocessing
    """
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    
    IMG_SIZE = 224
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    
    
    img_array = np.array(image, dtype=np.float32)
    
    
    img_array = img_array / 255.0  
    
   
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def validate_medical_image(image: Image.Image) -> bool:
    """
    Tıbbi görüntü doğrulaması
    """
    width, height = image.size
    
    
    if width < 100 or height < 100:
        return False
    
    
    aspect_ratio = width / height
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False
    
    return True

@app.post("/predict", response_model=PredictionResponse)
async def predict_covid(file: UploadFile = File(...)):
    """
    Geliştirilmiş COVID-19 tahmini
    """
    try:
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Sadece resim dosyaları kabul edilir")
        
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        
        if not validate_medical_image(image):
            raise HTTPException(
                status_code=400, 
                detail="Geçersiz görüntü boyutu veya formatı. Lütfen geçerli bir röntgen görüntüsü yükleyin."
            )
        
        
        processed_image = preprocess_image(image)
        
        
        predictions = model.predict(processed_image, verbose=0)
        predicted_prob = predictions[0]
        
        
        predicted_class_idx = np.argmax(predicted_prob)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predicted_prob[predicted_class_idx])
        
        
        MIN_CONFIDENCE = 0.5  
        if confidence < MIN_CONFIDENCE:
            predicted_class = "Belirsiz"
            confidence = 0.0
        
        
        probabilities = {
            CLASS_NAMES[i]: float(predicted_prob[i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=probabilities
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")


@app.get("/model-info")
async def get_model_info():
    """
    Model bilgilerini döndürür
    """
    try:
        
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        return {
            "input_shape": input_shape,
            "output_shape": output_shape,
            "total_params": model.count_params(),
            "class_names": CLASS_NAMES
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def main_page():
    """
    Ana sayfa - dosya yükleme formu
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>COVID-19 Tanı Sistemi</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .upload-area {
                border: 2px dashed #ddd;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                background-color: #fafafa;
            }
            .upload-area:hover {
                border-color: #007bff;
                background-color: #f0f8ff;
            }
            input[type="file"] {
                margin: 10px 0;
            }
            button {
                background-color: #007bff;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #0056b3;
            }
            button:disabled {
                background-color: #6c757d;
                cursor: not-allowed;
            }
            .result {
                margin-top: 20px;
                padding: 20px;
                border-radius: 5px;
                display: none;
            }
            .result.normal {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
            }
            .result.covid {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
            }
            .result.pneumonia {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                color: #856404;
            }
            .result.belirsiz {
                background-color: #e2e3e5;
                border: 1px solid #d6d8db;
                color: #383d41;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            #preview {
                max-width: 300px;
                max-height: 300px;
                margin: 20px auto;
                display: none;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .warning {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                color: #856404;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧪 COVID-19 Tanı Sistemi</h1>
            <p style="text-align: center; color: #666;">
                Akciğer röntgeni yükleyerek COVID-19, Pnömoni veya Normal durumu tespit edin
            </p>
            
            <div class="warning">
                ⚠️ <strong>Önemli:</strong> Bu sistem sadece yardımcı amaçlıdır. Kesin tanı için mutlaka doktorunuza başvurun.
            </div>
            
            <form id="uploadForm">
                <div class="upload-area">
                    <h3>📁 Röntgen Görüntüsü Yükleyin</h3>
                    <p style="font-size: 14px; color: #666;">
                        Desteklenen formatlar: JPG, PNG, JPEG<br>
                        Minimum boyut: 100x100 piksel
                    </p>
                    <input type="file" id="fileInput" accept="image/*" required>
                    <br><br>
                    <button type="submit" id="analyzeBtn">🔍 Analiz Et</button>
                </div>
            </form>
            
            <img id="preview" />
            
            <div class="loading" id="loading">
                <p>🔄 Görüntü analiz ediliyor, lütfen bekleyin...</p>
                <p style="font-size: 12px; color: #666;">Bu işlem 5-10 saniye sürebilir.</p>
            </div>
            
            <div class="result" id="result">
                <h3 id="prediction"></h3>
                <p><strong>Güven Oranı: </strong><span id="confidence"></span></p>
                <div id="probabilities"></div>
                <p style="font-size: 12px; color: #666; margin-top: 15px;">
                    ⚠️ Bu sonuç sadece yardımcı amaçlıdır. Kesin tanı için doktorunuza başvurun.
                </p>
            </div>
        </div>

        <script>
            document.getElementById('fileInput').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    // Dosya boyutu kontrolü (5MB)
                    if (file.size > 5 * 1024 * 1024) {
                        alert('Dosya boyutu çok büyük! Maksimum 5MB olmalı.');
                        this.value = '';
                        return;
                    }
                    
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const preview = document.getElementById('preview');
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                }
            });

            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                const analyzeBtn = document.getElementById('analyzeBtn');
                
                if (!file) {
                    alert('Lütfen bir dosya seçin!');
                    return;
                }
                
                // Loading göster
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = '🔄 Analiz Ediliyor...';
                
                // FormData oluştur
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    // API'ye gönder
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        // Sonuçları göster
                        displayResult(data);
                    } else {
                        alert('Hata: ' + data.detail);
                    }
                } catch (error) {
                    alert('Bağlantı hatası: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = '🔍 Analiz Et';
                }
            });
            
            function displayResult(data) {
                const resultDiv = document.getElementById('result');
                const predictionEl = document.getElementById('prediction');
                const confidenceEl = document.getElementById('confidence');
                const probabilitiesEl = document.getElementById('probabilities');
                
                // Tahmin sonucunu göster
                let emoji = '';
                let className = '';
                
                switch(data.prediction) {
                    case 'Normal':
                        emoji = '✅';
                        className = 'normal';
                        break;
                    case 'COVID-19':
                        emoji = '🦠';
                        className = 'covid';
                        break;
                    case 'Pneumonia':
                        emoji = '⚠️';
                        className = 'pneumonia';
                        break;
                    case 'Belirsiz':
                        emoji = '❓';
                        className = 'belirsiz';
                        break;
                }
                
                predictionEl.textContent = `${emoji} Sonuç: ${data.prediction}`;
                confidenceEl.textContent = `%${(data.confidence * 100).toFixed(1)}`;
                
                // Olasılıkları göster
                let probText = '<h4>Detaylı Olasılıklar:</h4><ul>';
                for (const [className, prob] of Object.entries(data.probabilities)) {
                    probText += `<li><strong>${className}:</strong> %${(prob * 100).toFixed(1)}</li>`;
                }
                probText += '</ul>';
                probabilitiesEl.innerHTML = probText;
                
                // Result div'ini göster ve class ekle
                resultDiv.className = `result ${className}`;
                resultDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "COVID-19 Tanı API'si çalışıyor"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)