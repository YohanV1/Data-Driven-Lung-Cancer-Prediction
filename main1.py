import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import manhattan_distances

@st.cache_resource
def load_model():
    with open('knn_with_pca.pkl', 'rb') as file:
        loaded_knn_model = pickle.load(file)

    return loaded_knn_model


st.set_page_config(layout="wide", page_title='Lung Cancer Staging and '
                                             'Prediction')


st.sidebar.title("Lung Cancer Staging and Prediction")
with st.sidebar.expander("About"):
    st.write("This application was built on the lung cancer dataset from "
             "[Kaggle](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link)."
             " Please don't hesitate to reach out.")
# with st.sidebar.expander('Model Metrics '):
#     st.subheader("Decision Tree: ")
#     st.write("Train = 84.06%, Test = 85.33%")
#     st.subheader("Random Forest: ")
#     st.write("Train = 93.05%, Test = 89.67%")
#     st.subheader("XGBoost: ")
#     st.write("Train = 95.78%, Test = 90.76%")

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
            'Coughing of Blood': coughing_blood,
            'Passive Smoker': passive_smoker,
            'Obesity': obesity,
            'Dust Allergy': dust_allergy,
            'Genetic Risk': genetic_risk
        }
        df = pd.DataFrame(data)

        prediction = knn_model.predict(df.iloc[0:1])

        st.subheader(f"Prediction: {prediction}")
        # if prediction[0] == 0:

        #     st.success('Patient does not have heart disease. '
        #                '[Prediction from XGBoost]')
        # if prediction[0] == 1:
        #     st.error('Patient has heart disease.'
        #              '[Prediction from XGBoost]')