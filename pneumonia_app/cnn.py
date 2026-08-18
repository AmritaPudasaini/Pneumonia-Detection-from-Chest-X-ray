import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)

model = load_model("cnn_pneumonia_model.h5")

def preprocess_image(image_bytes):

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

app = FastAPI(
    title="Pneumonia Detection API",
    description="Upload a chest X-ray image to classify it as Normal or Pneumonia using a trained CNN model.",
    version="1.0.0"
)

@app.post(
    "/predict",
    summary="Classify a chest X-ray image",
    description="Upload a chest X-ray image (JPG/PNG). Returns whether the image shows Pneumonia or Normal, along with a confidence percentage.",
    response_description="Prediction result with confidence score"
)

async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    x = preprocess_image(image_bytes)
    pred_prob = model.predict(x)[0][0]
    pred_label = "PNEUMONIA" if pred_prob > 0.5 else "NORMAL"
    confidence = float(pred_prob) if pred_label == "PNEUMONIA" else float(1 - pred_prob)
    return {
        "prediction": pred_label,
        "confidence": round(confidence * 100, 2)
    }