from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# Load the trained model and encoders
model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

app = FastAPI()

class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.post("/predict/")
async def predict(customer_data: CustomerData):
    input_data = pd.DataFrame([customer_data.dict()])

    # One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    
    # Concatenate the encoded Geography with the input DataFrame
    input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    
    # Drop the original Geography column
    input_df = input_df.drop(columns=['Geography'])

    # Scale the input data
    input_data_scaled = scaler.transform(input_df)

    # Prediction
    churn_pred = model.predict(input_data_scaled)
    churn_pred_proba = churn_pred[0][0]

    return {"churn_probability": churn_pred_proba, "likely_to_churn": churn_pred_proba > 0.5}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
