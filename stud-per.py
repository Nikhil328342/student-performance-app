import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_model():
    with open("student_lr_final_model.pkl", 'rb') as file:
        model,scaler,le = pickle.load(file)
    return model,scaler,le

def preprocessing_input_data(data,scaler,le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocessing_input_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction


def main():
    st.title("Student Performance Prediction")
    st.write("Enter your data to get a prediction for your performance")

    hour_studied = st.number_input("Hours Studied", min_value=1, max_value=10, value=5)
    previous_score = st.number_input("Previous Score", min_value=30, max_value=100, value=50)
    Extra_curriculam = st.selectbox("extra curriculam activity", ["Yes", "No"])
    Sleep_hour = st.number_input("Sleeping Hours", min_value=4, max_value=9, value=6)
    Paper_solved = st.number_input("Number of Question Paper Solved", min_value=0, max_value=10, value=5)

    if st.button("Predict-Your_Score"):
        user_data = {
            "Hours Studied" : hour_studied,
            "Previous Scores" : previous_score,
            "Extracurricular Activities" : Extra_curriculam,
            "Sleep Hours" : Sleep_hour,
            "Sample Question Papers Practiced" : Paper_solved
        }
        prediction = predict_data(user_data)
        st.success(f"Your prediction result is {prediction}")


if __name__ == "__main__" :
    main()