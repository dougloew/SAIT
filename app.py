# Filtering Warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model

model = load_model('insurance_dt_model')

#input_dict = {'age':20, 'sex':'male', 'bmi':20, 'children':2, 'smoker':'yes', 'region':'southwest'}
input_dict = {'age':20, 'sex':'female', 'bmi':20, 'children':2, 'smoker':'no', 'region':'southwest'}

input_df = pd.DataFrame([input_dict])
#print(input_df)

predictions_df = predict_model(estimator=model, data=input_df)
#print(predictions_df)

predictions = predictions_df.iloc[0]['prediction_label']
#print(predictions)

st.markdown(predictions)


st.header('Insurance Charges')

