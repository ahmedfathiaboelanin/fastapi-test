# main.py
# uvicorn main:app --reload
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sklearn.preprocessing import OneHotEncoder
import joblib
import traceback
from pydantic import BaseModel
import numpy as np

# Load your original data
original_data = pd.read_csv("original_data.csv")

# Extract unique symptom values from the columns containing symptoms
symptom_columns = original_data.columns[1:]  # Assuming symptom columns start from the second column
all_possible_symptoms = original_data[symptom_columns].stack().unique().tolist()

# Create a one-hot encoder for the symptoms
encoder = OneHotEncoder(categories=[all_possible_symptoms], sparse=False)

# Fit the encoder with all possible symptom values
encoder.fit([[symptom] for symptom in all_possible_symptoms])

# Load your pre-trained model
model_path = os.path.join(os.getcwd(), "model.pkl")
model = joblib.load(model_path)

# Initialize FastAPI
app = FastAPI()

# Define a Pydantic model for request body
class Symptoms(BaseModel):
    symptoms: list

# Define the prediction endpoint with @app.post
@app.post("/predict")
async def predict(symptoms: Symptoms):
    try:
        print(f"Received symptoms: {symptoms}")

        # Assuming your model takes a list of symptoms and returns a prediction
        symptoms_list = symptoms.symptoms

        # Use one-hot encoding to convert string symptoms to numeric values
        symptoms_encoded = encoder.transform([[symptom] for symptom in symptoms_list])

        print(f"Encoded symptoms: {symptoms_encoded}")

        # Make predictions using the pre-trained model
        predictions = model.predict(symptoms_encoded)

        print(f"Predictions: {predictions}")

        # Return unique predictions
        unique_predictions = list(np.unique(predictions))

        print(f"Unique Predictions: {unique_predictions}")

        # Return the unique predictions as JSON
        return {"diagnosis": unique_predictions}
    except Exception as e:
        # Log the detailed error information for debugging
        traceback.print_exc()

        # Include the exception details in the response for better debugging
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal Server Error: {str(e)}"},
        )
