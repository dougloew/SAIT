# Filtering Warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model

model = load_model('insurance_dt_model')



#input_dict = {'age':20, 'sex':'male', 'bmi':20, 'children':2, 'smoker':'yes', 'region':'southwest'}
#input_dict = {'age':20, 'sex':'female', 'bmi':20, 'children':2, 'smoker':'no', 'region':'southwest'}

#input_df = pd.DataFrame([input_dict])
#print(input_df)

#predictions_df = predict_model(estimator=model, data=input_df)
#print(predictions_df)

#predictions = predictions_df.iloc[0]['prediction_label']
#print(predictions)




# Streamlit app
st.header('Insurance Charges Prediction')

# Input fields
age = st.number_input('Age', min_value=0, max_value=100, value=20)
sex = st.selectbox('Sex', options=['male', 'female'])
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=20.0)
children = st.number_input('Children', min_value=0, max_value=10, value=2)
smoker = st.selectbox('Smoker', options=['yes', 'no'])
region = st.selectbox('Region', options=['southwest', 'southeast', 'northwest', 'northeast'])

# Create input dictionary
input_dict = {
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'children': children,
    'smoker': smoker,
    'region': region
}

# Convert input dictionary to DataFrame
input_df = pd.DataFrame([input_dict])


# Make predictions
predictions_df = predict_model(estimator=model, data=input_df)
predictions = predictions_df.iloc[0]['prediction_label']

# Display predictions
st.markdown(f"**Predicted Insurance Charges:** {predictions}")