import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Load the saved model and label encoders
model_data = joblib.load('vegetable_price_predictor_with_encoders.pkl')

# Extract the model and label encoders from the dictionary
model = model_data['model']
label_encoders = model_data['label_encoders']

# Define a reference arrival date (or find the minimum from your data)
reference_arrival_date = pd.to_datetime("2025-01-01")  # Set a reference date, or pick the minimum date from your dataset

# Function for price prediction
def predict_price(state, district, market, commodity, variety, grade, arrival_date, min_price, max_price):
    encoded_features = []
    for feature, value in zip(["State", "District", "Market", "Commodity", "Variety", "Grade"], 
                              [state, district, market, commodity, variety, grade]):
        if value in label_encoders[feature].classes_:
            encoded_features.append(label_encoders[feature].transform([value])[0])
        else:
            encoded_features.append(-1)
    
    # Calculate the difference in days between the given arrival date and the reference arrival date
    encoded_features.append((pd.to_datetime(arrival_date) - reference_arrival_date).days)  # Convert date
    encoded_features.extend([min_price, max_price])
    
    feature_columns = ["State", "District", "Market", "Commodity", "Variety", "Grade", "Arrival Date", "Min Price", "Max Price"]
    X_new = pd.DataFrame([encoded_features], columns=feature_columns)
    
    predicted_price = model.predict(X_new)[0]
    
    return round(predicted_price, 2)

# Streamlit User Interface
st.title('Vegetable Price Prediction')
st.markdown('Enter the details below to predict the price of the vegetable.')

# Input fields for user to enter
state = st.text_input('State')
district = st.text_input('District')
market = st.text_input('Market')
commodity = st.text_input('Commodity')
variety = st.text_input('Variety')
grade = st.text_input('Grade')
arrival_date = st.date_input('Arrival Date')
min_price = st.number_input('Min Price', min_value=0.0, step=0.1)
max_price = st.number_input('Max Price', min_value=0.0, step=0.1)

# Predict button
if st.button('Predict Price'):
    predicted_price = predict_price(state, district, market, commodity, variety, grade, str(arrival_date), min_price, max_price)
    st.write(f"Predicted Price: ₹{predicted_price}")
