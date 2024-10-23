from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("./saved-models/model.keras", compile=False)
AUTOENCODER = tf.keras.models.load_model("./saved-models/autoencoder.keras", compile=False)
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def compute_reconstruction_error(original, reconstructed):
    return np.mean(np.square(reconstructed - original))

def preprocess_image(img: np.ndarray) -> np.ndarray:
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def is_potato_leaf(image: np.ndarray, threshold=0.034) -> bool:
    preprocessed_image = preprocess_image(image)
    reconstructed_img = AUTOENCODER.predict(preprocessed_image)
    error = compute_reconstruction_error(preprocessed_image, reconstructed_img)
    print(f"Reconstruction error: {error}")
    
    if error >= threshold:
        return False
    return True

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert('RGB')
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return image

def read_file_and_resize(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert('RGB')
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_file = await file.read()
    image = read_file_as_image(temp_file)
    # Due to the autoencoder waiting a long time to train if set more than 64x64, I need to resize the image to 64x64
    im = read_file_and_resize(temp_file)

    if not is_potato_leaf(im):
        return {"class": "Not a potato leaf", "confidence": None}
    
    img_batch = np.expand_dims(image, axis=0)
    predictions = MODEL.predict(img_batch)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        'class': predicted_class,
        'confidence': confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)