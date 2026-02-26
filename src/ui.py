import streamlit as st
import requests

# =========================
# STREAMLIT APP
# =========================

st.title("Air Quality Prediction")

st.write("Masukkan data polutan untuk prediksi kualitas udara.")

with st.form("air_form"):
    stasiun = st.selectbox(
    "Stasiun",
    [
        "DKI1 (Bunderan HI)",
        "DKI2 (Kelapa Gading)",
        "DKI3 (Jagakarsa)",
        "DKI4 (Lubang Buaya)",
        "DKI5 (Kebon Jeruk) Jakarta Barat"
    ]
)
    pm10 = st.number_input("PM10", min_value=0)
    pm25 = st.number_input("PM25", min_value=0)
    so2 = st.number_input("SO2", min_value=0)
    co = st.number_input("CO", min_value=0)
    o3 = st.number_input("O3", min_value=0)
    no2 = st.number_input("NO2", min_value=0)

    submit = st.form_submit_button("Predict")

if submit:
    payload = {
        "stasiun": stasiun,
        "pm10": int(pm10),
        "pm25": int(pm25),
        "so2": int(so2),
        "co": int(co),
        "o3": int(o3),
        "no2": int(no2)
    }

    with st.spinner("Mengirim ke server..."):
        response = requests.post(
            "http://127.0.0.1:8080/predict",
            json=payload
        )

    result = response.json()

    if result["error_msg"] != "":
        st.error(result["error_msg"])
    else:
        st.success(f"Hasil Prediksi: {result['res']}")

