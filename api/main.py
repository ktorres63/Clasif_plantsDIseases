from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2


app = FastAPI()
MODEL = tf.keras.models.load_model("../training/saved_models/v1")
CLASS_NAMES = ["Enferma", "Medianamente Enferma", "Saludable"]


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = read_file_as_image(await file.read())
    image =cv2.resize(img,(256,256))
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'Prediccion':predicted_class,
        'confianza': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app,host="localhost", port=8000)

