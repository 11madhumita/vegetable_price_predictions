import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder

# Load model and label encoders
model_data = joblib.load('vegetable_price_predictor_with_encoders.pkl')
model = model_data['model']
label_encoders = model_data['label_encoders']

# Load dataset for dropdown options
data = pd.read_csv('commodity_prices.csv')

# Extract dropdown values
states = sorted(data['State'].dropna().unique())
districts = sorted(data['District'].dropna().unique())
markets = sorted(data['Market'].dropna().unique())
commodities = sorted(data['Commodity'].dropna().unique())
varieties = sorted(data['Variety'].dropna().unique())
grades = sorted(data['Grade'].dropna().unique())

# Set reference date for arrival date processing
reference_arrival_date = pd.to_datetime("2025-01-01")

# Prediction function
def predict_price(state, district, market, commodity, variety, grade, arrival_date, min_price, max_price):
    encoded_features = []
    for feature, value in zip(
        ["State", "District", "Market", "Commodity", "Variety", "Grade"],
        [state, district, market, commodity, variety, grade]
    ):
        if value in label_encoders[feature].classes_:
            encoded_features.append(label_encoders[feature].transform([value])[0])
        else:
            encoded_features.append(-1)

    date_diff = (pd.to_datetime(arrival_date) - reference_arrival_date).days
    encoded_features.append(date_diff)
    encoded_features.extend([min_price, max_price])

    feature_columns = ["State", "District", "Market", "Commodity", "Variety", "Grade", "Arrival Date", "Min Price", "Max Price"]
    X_new = pd.DataFrame([encoded_features], columns=feature_columns)

    predicted_price = model.predict(X_new)[0]
    return round(predicted_price, 2)

# Streamlit UI
st.title("🥦 Vegetable Price Predictor")
st.markdown("Provide details below to predict the **vegetable selling price**.")

# Dropdown inputs
state = st.selectbox("State", states)
district = st.selectbox("District", districts)
market = st.selectbox("Market", markets)
commodity = st.selectbox("Commodity", commodities)
variety = st.selectbox("Variety", varieties)
grade = st.selectbox("Grade", grades)
arrival_date = st.date_input("Arrival Date")
min_price = st.number_input("Min Price (₹)", min_value=0.0, step=0.5)
max_price = st.number_input("Max Price (₹)", min_value=0.0, step=0.5)

# Prediction trigger
if st.button("Predict Price"):
    predicted_price = predict_price(
        state, district, market, commodity, variety, grade,
        str(arrival_date), min_price, max_price
    )
    st.success(f"📈 Predicted Price: ₹{predicted_price}")
