from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageFile
import io

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_INPUT_SIZE = (180, 180)

model = tf.keras.models.load_model("bone_fracture_model.h5")

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = ImageOps.fit(image, MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)
    img = np.asarray(image).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    img = preprocess_image(image)
    prediction = model.predict(img)
    score = float(prediction[0][0])

    result = "Normal" if score > 0.5 else "Fracture Detected"

    return {
        "score": round(score, 4),
        "result": result
    }
