import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Tourism Package Prediction", page_icon="✈️")

# Load the model committed by the pipeline (sits next to this file)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_package_model_v1.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Same threshold that was used in train.py
CLASSIFICATION_THRESHOLD = 0.45

st.title("Wellness Tourism Package Prediction")
st.write("Fill in the customer details below to predict whether they will purchase the package.")

col1, col2 = st.columns(2)
with col1:
    Age = st.slider("Age", 18, 70, 30)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Unmarried", "Divorced"])
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=23000.0, step=500.0)
    Passport = st.selectbox("Has Passport?", ["Yes", "No"])
with col2:
    NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 3)
    NumberOfChildrenVisiting = st.slider("Number of Children (under 5) Visiting", 0, 3, 1)
    NumberOfTrips = st.slider("Average Trips per Year", 1, 22, 3)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    DurationOfPitch = st.slider("Duration of Pitch (mins)", 5, 130, 15)
    NumberOfFollowups = st.slider("Number of Follow-ups", 1, 6, 4)
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

# Assemble the inputs in the same raw format used for training
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

if st.button("Predict"):
    prob = float(model.predict_proba(input_data)[0, 1])
    pred = int(prob >= CLASSIFICATION_THRESHOLD)
    st.metric("Purchase probability", f"{prob:.1%}")
    if pred == 1:
        st.success("Prediction: the customer is **likely to purchase** the Wellness Tourism Package.")
    else:
        st.info("Prediction: the customer is **unlikely to purchase** the package.")
    with st.expander("Input sent to the model"):
        st.dataframe(input_data)
