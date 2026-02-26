# ======================================
# AIR QUALITY PREDICTION - FASTAPI
# ======================================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src import utils
from src import data_pipeline
from src import preprocessing


# ======================================
# CONFIGURATION
# ======================================

PATH_CONFIG = "config/config.yaml"

# Load configuration
config = utils.load_config(PATH_CONFIG)

# Load serialized objects
ohe_stasiun = utils.deserialize_data(config["path_fitted_encoder_stasiun"])
scaler = utils.deserialize_data(config["path_fitted_scaler"])
le_encoder = utils.deserialize_data(config["path_fitted_encoder_label"])
best_model = utils.deserialize_data(config["path_production_model"])


# ======================================
# INPUT DATA STRUCTURE
# ======================================

class DataAPI(BaseModel):
    stasiun: str
    pm10: int
    pm25: int
    so2: int
    co: int
    o3: int
    no2: int


# ======================================
# CREATE FASTAPI APP
# ======================================

app = FastAPI(
    title="Air Quality Prediction API",
    description="Predict air quality category based on pollutant levels",
    version="1.0.0"
)


# ======================================
# ROUTES
# ======================================

@app.get("/")
def home():
    return {"message": "Air Quality API is running "}


@app.post("/predict")
def predict(data: DataAPI):

    # Convert Pydantic object to DataFrame
    data = pd.DataFrame([data.dict()])

    # Ensure correct column order
    data = data[config["features"]]

    # Data defense (range checking)
    try:
        data_pipeline.data_defense(data, config, api=True)
    except AssertionError as err:
        return {
            "res": None,
            "error_msg": str(err)
        }

    # One-Hot Encoding (stasiun)
    data = preprocessing.transform_ohe_encoder(data, ohe_stasiun)

    # Scaling numeric features
    data = preprocessing.transform_scaler(data, scaler)

    # Predict
    y_pred = best_model.predict(data)

    # Decode predicted label
    y_pred = le_encoder.inverse_transform(y_pred)[0]

    return {
        "res": y_pred,
        "error_msg": ""
    }


# ======================================
# OPTIONAL: RUN VIA PYTHON DIRECTLY
# ======================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )
