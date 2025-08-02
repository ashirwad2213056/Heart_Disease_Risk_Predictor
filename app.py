import streamlit as st 
import pandas as pd 
import joblib

model_knn = joblib.load('KNNHeart.pkl')
model_log = joblib.load('LogisticHeart.pkl')
scaler = joblib.load('scaler.pkl')  
expectedColumns = joblib.load('columns.pkl') 

st.title("Heart Disease Prediction App by Ashirwad Gupta")    
st.markdown("Provide the required details below to predict heart disease.")

# Input fields
Age = st.slider("Age", 18, 100, 30)
Sex = st.selectbox("Sex", ['M', 'F'])
ChestPain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'ASY', 'TA'])
RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
Cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
RestingECG = st.selectbox("Resting Electrocardiographic Results", ['Normal', 'ST', 'LVH'])
MaxHR = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
ExerciseAngina = st.selectbox("Exercise Induced Angina", ['Y', 'N'])
Oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
ST_Slope = st.selectbox("Slope of ST Segment", ['Up', 'Flat', 'Down'])

# Predict
if st.button("Predict"):
    rawInput = {
        'Age': Age,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'MaxHR': MaxHR,
        'Oldpeak': Oldpeak,
        'FastingBS': FastingBS,
        'Sex_' + Sex: 1,
        'ChestPainType_' + ChestPain: 1,
        'RestingECG_' + RestingECG: 1,
        'ExerciseAngina_' + ExerciseAngina: 1,
        'ST_Slope_' + ST_Slope: 1
    }

    inputData = pd.DataFrame([rawInput])

    # Fill in missing columns
    for col in expectedColumns:
        if col not in inputData.columns:
            inputData[col] = 0

    inputData = inputData[expectedColumns]

    # Scale input
    scaledInput = scaler.transform(inputData)

    # Predict using both models
    predictionKNN = model_knn.predict(scaledInput)[0]
    predictionLogistic = model_log.predict(scaledInput)[0]

    # Output predictions
    if predictionKNN == 1:
        st.error("KNN: High risk of heart disease.")
    else:
        st.success("KNN: Low risk of heart disease.")

    if predictionLogistic == 1:
        st.error("Logistic Regression: High chances of heart disease.")
    else:
        st.success("Logistic Regression: Low chances of heart disease.")
