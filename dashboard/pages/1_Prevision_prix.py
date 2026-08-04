import os

import requests
import streamlit as st

st.set_page_config(page_title="GetAround - Prévision de prix", layout="wide")

st.markdown(
    "<style>.block-container { padding-top: 2rem; }</style>",
    unsafe_allow_html=True,
)

API_URL = os.environ.get("PRICING_API_URL", "http://15.224.83.177:8000/predict")

MODEL_KEYS = [
    "Citroën", "Renault", "BMW", "Peugeot", "Audi", "Nissan", "Mitsubishi", "Mercedes",
    "Volkswagen", "Toyota", "SEAT", "Subaru", "Opel", "Ferrari", "PGO", "Maserati",
    "Suzuki", "Porsche", "Ford", "KIA Motors", "Alfa Romeo", "Fiat", "Lexus",
    "Lamborghini", "Mini", "Mazda", "Honda", "Yamaha",
]
FUELS = ["diesel", "petrol", "hybrid_petrol", "electro"]
PAINT_COLORS = ["black", "grey", "blue", "white", "brown", "silver", "red", "beige", "green", "orange"]
CAR_TYPES = ["estate", "sedan", "suv", "hatchback", "subcompact", "coupe", "convertible", "van"]

st.title("Prévision de prix de location")
st.markdown("Renseigne les caractéristiques du véhicule pour estimer son prix de location journalier.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        model_key = st.selectbox("Marque (model_key)", MODEL_KEYS)
        mileage = st.number_input("Kilométrage (mileage)", min_value=0, value=100000, step=1000)
        engine_power = st.number_input("Puissance moteur (engine_power)", min_value=0, value=120, step=10)

    with col2:
        fuel = st.selectbox("Carburant (fuel)", FUELS)
        paint_color = st.selectbox("Couleur (paint_color)", PAINT_COLORS)
        car_type = st.selectbox("Type de véhicule (car_type)", CAR_TYPES)

    with col3:
        private_parking_available = st.checkbox("Parking privé disponible")
        has_gps = st.checkbox("GPS")
        has_air_conditioning = st.checkbox("Climatisation")
        automatic_car = st.checkbox("Boîte automatique")
        has_getaround_connect = st.checkbox("GetAround Connect")
        has_speed_regulator = st.checkbox("Régulateur de vitesse")
        winter_tires = st.checkbox("Pneus hiver")

    submitted = st.form_submit_button("Estimer le prix", type="primary")

if submitted:
    payload = {
        "input": [[
            model_key, mileage, engine_power, fuel, paint_color, car_type,
            private_parking_available, has_gps, has_air_conditioning, automatic_car,
            has_getaround_connect, has_speed_regulator, winter_tires,
        ]]
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        price = response.json()["prediction"][0]

        with st.container(border=True):
            st.markdown("##### 💰 Prix de location journalier estimé")
            st.markdown(
                f"<h1 style='text-align:center;'>{price:.0f} € / jour</h1>",
                unsafe_allow_html=True,
            )
    except requests.exceptions.RequestException as e:
        st.error(f"Impossible de contacter l'API de prédiction ({API_URL}) : {e}")
