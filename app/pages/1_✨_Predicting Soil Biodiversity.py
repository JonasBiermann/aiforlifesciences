# vinson you can add your graphics here. Remember the format is 
# title - description - graphic with alternating designs of 2 graphics and 1 graphic

import streamlit as st
from dataframe import *

import catboost as cb
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt

import numpy as np

from helper.visualization import (
    beeswarm_plot,
    prediction_distribution_plot, 
    confusion_matrix_plot,
    threshold_performance_plot,
    shap_pdp_plot,
    roc_auc_plot)

shap.initjs()

path = os.path.dirname(__file__)

st.set_page_config(page_title="Predicting Soil Biodiversity", page_icon="✨", layout='wide')

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

catboost_model_path = path+'/../artifacts/CatBoostShannon.cb'

model = cb.CatBoostRegressor(loss_function='RMSE')
model.load_model(catboost_model_path)
explainer = shap.TreeExplainer( # I make a mistake here, this should be loaded instead of calculate on the fly like this
    model
)
list_X = ['pH_CaCl2','pH_H2O','CaCO3','EC','OC','P','N','K','LC0_Desc','LC1_Desc','LU1_Desc']
list_X_cat = ['LC0_Desc','LC1_Desc','LU1_Desc']
list_X_num = [x for x in list_X if x not in list_X_cat]
with st.expander("Input your data") : 
    with st.form('Prediction'):
        input_data = {}

        cols_numeric= st.columns(len(list_X_num))

        def create_input_field(column, label):
            val = column.number_input(label, min_value=min(df[label]), max_value=max(df[label]), value=df[label].mean())
            return val

        for col in zip(cols_numeric,list_X_num):
            val = create_input_field(col[0], col[1])
            input_data[col[1]] = round(val,2)


        cols_categorical= st.columns(len(list_X_cat))
        for col,label in zip(cols_categorical,list_X_cat):
            val =  st.selectbox(
                label,
                df[label].unique(),
                    placeholder=df[label].iloc[0],
                    )
            input_data[label] = val
        
        clicked = st.form_submit_button('Predict')

    if clicked:
        input_data = pd.Series(input_data)
        result = model.predict(input_data)
        st.metric('The Shannon Index of this soil sample predicted to be',round(result,5))
        single_shap = explainer(input_data.to_frame(0).transpose())
   
        # Generate a force plot without specifying link='logit'
        fig = shap.plots.force(single_shap[0],show=False)
        st.pyplot(fig.matplotlib(figsize=[15,6],show=False,text_rotation=0))


    

