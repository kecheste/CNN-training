# Potato Leaf Classification API

This is a FastAPI application that classifies potato leaf diseases and can also detect if the image provided is **not** a potato leaf using an autoencoder. The classifier identifies three types of potato leaf conditions: 

1. Potato___Early_blight
2. Potato___Late_blight
3. Potato___healthy

If the image is not a potato leaf, the API returns "Not a potato leaf".

## Features

- **Autoencoder-based anomaly detection**: Detects if an image is not a potato leaf.
- **Classification model**: Classifies potato leaves into one of three categories.
- **REST API**: Upload an image and get the prediction.

## I followed two completely different approaches to solve the problem of when a model is provided another image completely different than potato leaf and tries to classify it, the model will try to classify the image among the three classes based on similarity. To fix this issue

1. I added another class to the dataset, "Non_potato_leaf" and provided randon unsplash images. Then I retrained the original model which classifies the potato leaves. Therefore finally when the model is provided with unknown picture it will most probabily predict it as non potato leaf/image.

- The codelab for this approach can be found in **/training/with_additional_class.ipynb** and the model created is found in **/saved-models/additional_class.keras**

2. I used an autoencoder to reconstruct the input as accurately as possible. By comparing the training dataset with its reconstruction, we can measure the reconstruction error.

- The codelab for this approach can be found in **/training/Autoencoder.ipynb** and the autoencoder model is found in **/saved-models/autoencoder**


### To run this application, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/kecheste/CNN-training.git
    cd CNN-training
    ```

2. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the FastAPI server**:
    ```bash
    uvicorn api.main:app --reload  # For the classification
    uvicorn api.main_autoencoder:app --reload  # For the autoencoder
    ```

5. **Test the API by sending an image in a post request**:
    Open your api client and go to `http://127.0.0.1:8000/predict`

### Example Response

When you upload an image to the API, you will receive a JSON response with the classification result. Here is an example response for each approach:

#### Using the Classification Model:
```json
{
    "prediction": "Potato___Early_blight",
    "confidence": 0.995
}
```

#### Using the Autoencoder:
```json
{
    "prediction": "Not a potato leaf",
    "reconstruction_error": 0.15
}
```

