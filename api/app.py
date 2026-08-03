import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "model/model.joblib"
COLUMNS = [
    "model_key",
    "mileage",
    "engine_power",
    "fuel",
    "paint_color",
    "car_type",
    "private_parking_available",
    "has_gps",
    "has_air_conditioning",
    "automatic_car",
    "has_getaround_connect",
    "has_speed_regulator",
    "winter_tires",
]

app = FastAPI(
    title="GetAround Pricing API",
    description="API de prédiction du prix de location journalier d'un véhicule.",
)

pipeline = joblib.load(MODEL_PATH)


class PredictionInput(BaseModel):
    input: list[list]


@app.post(
    "/predict",
    summary="Prédit le prix de location journalier",
    description=(
        "Prend en entrée une liste de véhicules (chaque véhicule étant une liste de "
        f"{len(COLUMNS)} valeurs dans l'ordre : {', '.join(COLUMNS)}) et retourne le "
        "prix de location journalier prédit pour chacun.\n\n"
        "Exemple d'entrée :\n"
        '```json\n{"input": [["Citroën", 140411, 100, "diesel", "black", '
        '"convertible", true, true, false, false, true, true, true]]}\n```\n\n'
        "Exemple de sortie :\n"
        '```json\n{"prediction": [106.0]}\n```'
    ),
)
def predict(data: PredictionInput):
    try:
        df = pd.DataFrame(data.input, columns=COLUMNS)
        prediction = pipeline.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"prediction": prediction.tolist()}
