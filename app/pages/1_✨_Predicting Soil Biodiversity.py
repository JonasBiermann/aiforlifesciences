# vinson you can add your graphics here. Remember the format is 
# title - description - graphic with alternating designs of 2 graphics and 1 graphic

import streamlit as st
from dataframe import *

st.set_page_config(page_title="Predicting Soil Biodiversity", page_icon="✨")

st.markdown("# Predicting Soil Biodiversity with Machine Learning")

st.subheader("Description")
st.write("In our collaborative effort, we developed a robust machine learning model aimed at predicting soil biodiversity based on key soil characteristics. Leveraging the SHAP (SHapley Additive exPlanations) algorithm and classification techniques, our model draws insights from the LUCAS dataset, a comprehensive soil database.")
st.write("If you're interested about the detalis of this model just click Learn More!")
with st.expander('Learn More'):
    st.subheader("Data Selection")
    st.write("Our model is trained on the LUCAS dataset, which provides a rich source of soil information. The dataset includes crucial parameters such as pH, nitrogen (N), phosphorus (P), potassium (K), organic carbon (OC), and electrical conductivity (EC).")

    st.subheader("Feature Exploration")
    st.write("We meticulously explored the relationships between soil biodiversity and the selected features. pH levels, nutrient content (N, P, K), organic carbon, and electrical conductivity were identified as critical factors influencing soil biodiversity.")

    st.subheader("SHAP Algorithm")
    st.write("To unravel the intricate connections between features and biodiversity, we employed the SHAP algorithm. SHAP values offer insightful explanations for individual predictions, shedding light on the contribution of each feature to the model's output. This interpretability is crucial for understanding the model's decision-making process.")

    st.subheader("Classification")
    st.write("Our machine learning model employs a classification approach to predict soil biodiversity. By leveraging patterns within the LUCAS dataset, the model has learned to categorize soil samples into distinct biodiversity classes based on the specified features.")

    st.subheader("Validation and Fine-Tuning")
    st.write("We rigorously validated and fine-tuned the model to ensure its accuracy and generalizability. This involved splitting the dataset into training and testing sets, optimizing hyperparameters, and employing techniques to mitigate overfitting.")

    st.subheader("Results and Implications")
    st.write("The resulting model provides a reliable tool for predicting soil biodiversity, offering valuable insights for land management and conservation efforts. By understanding the influence of pH, nutrient levels, and other factors, our model contributes to a deeper understanding of soil ecosystems and supports informed decision-making in agriculture and environmental science. You can try it out yourself on the page Soil Data Check!")