import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

@st.cache_resource
def load_model():
    with open('knn_with_pca.pkl', 'rb') as file:
        loaded_knn_model = pickle.load(file)

    return loaded_knn_model


with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


st.set_page_config(layout="wide", page_title='Lung Cancer Staging and '
                                             'Prediction')


st.sidebar.title("Lung Cancer Staging and Prediction")
with st.sidebar.expander("About"):
    st.write("This application was built on the lung cancer dataset from "
             "[Kaggle](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link)."
             " Please don't hesitate to reach out.")
# with st.sidebar.expander('Model Metrics '):

st.sidebar.write('Parameters can be tuned further for better results.')
knn_model = load_model()

st.header('Lung Cancer Staging and Prediction - KNN Without PCA (Principal Component Analysis')

col1, ecol, col2 = st.columns([1.5, 0.2, 1.5])

with col1:
    coughing_blood = st.text_input('Coughing of Blood (1-9): ')
    passive_smoker = st.text_input('Passive Smoking Level (1-9): ')
    obesity = st.text_input('Obesity Level (1-9): ')
    dust_allergy = st.text_input('Dust Allergy Level (1-9): ')
    genetic_risk = st.text_input('Genetic Risk Level (1-9): ')

b = st.button("Run Model.")

if b:
    data = {
        'Coughing of Blood': [int(coughing_blood)],
        'Passive Smoker': [int(passive_smoker)],
        'Obesity': [int(obesity)],
        'Dust Allergy': [int(dust_allergy)],
        'Genetic Risk': [int(genetic_risk)]
    }
    print(data)

    df_data = pd.DataFrame(data)

    # Scale the user input
    xs = scaler.transform(df_data)

    df_xs = pd.DataFrame(xs, columns=df_data.columns)
    print(xs)
    # Make prediction using KNN model
    prediction = knn_model.predict(df_xs)
    st.subheader(f"Prediction: {prediction}")


