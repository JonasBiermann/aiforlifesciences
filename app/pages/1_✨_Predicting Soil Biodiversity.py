# vinson you can add your graphics here. Remember the format is 
# title - description - graphic with alternating designs of 2 graphics and 1 graphic

import streamlit as st
from dataframe import *

import catboost as cb
import shap
from streamlit_shap import st_shap
import joblib
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

model = cb.CatBoostClassifier() 
model.load_model("artifacts/CatboostRegressorV0demo.cbm")
explainer = shap.TreeExplainer( # I make a mistake here, this should be loaded instead of calculate on the fly like this
    model
)
shap_values = joblib.load("artifacts/SHAP_ExplainerObject")

# st.write(f"Probability of soil concentration have high CaCO3 concentration {result}")
list_X = ['pH_CaCl2','pH_H2O','EC','OC','P','N','K','LC0_Desc','LC1_Desc','LU1_Desc']
list_X_cat = ['LC0_Desc','LC1_Desc','LU1_Desc']
list_X_num = [x for x in list_X if x not in list_X_cat]
df_shap = pd.DataFrame(shap_values.values,columns=list_X)
df_shap.reset_index(inplace=True)

df_raw = pd.DataFrame(shap_values.data,columns=list_X)

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
        result = model.predict_proba(input_data)[1]
        st.write(f"Probability of the soil having high concentration of CaCO3 is : {result:.5f}")

        single_shap = explainer(input_data.to_frame(0).transpose())
   
        # Generate a force plot without specifying link='logit'
        fig = shap.plots.force(single_shap[0],show=False)
        st.pyplot(fig.matplotlib(figsize=None,show=False,text_rotation=0))


with st.expander('Model Analysis - SHAP'):
    st.header("SHAP Analysis")
    st.subheader("Global Analysis")
    st.write("Some explanation about the SHAP values")
    
    with st.container(border=True):
        fig_beeswarm,_ = beeswarm_plot(df_raw,df_shap,list_X_num,list_X_cat)
        st.pyplot(fig_beeswarm)

    st.subheader("Partial Dependence Plot")
    tabs = st.tabs(shap_values.feature_names)
    for tab,feature_name in zip(tabs,shap_values.feature_names) :
        fig_pdp = shap_pdp_plot(shap_values,feature_name,list_X_cat)
        tab.pyplot(fig_pdp)

with st.expander('Model Performance Evaluation'):
    st.header("Model Performance")

    with st.container(border=True):
        row1left, row1right = st.columns(2)

        row1left.subheader("Confusion Matrix")
        row1left.write("How many false positive and false negative ?")
        row1left.write("")
        fig_confmat = confusion_matrix_plot()
        row1left.pyplot(fig_confmat)

        row1right.subheader("Prediction Distribution")
        row1right.write("If the ground truth is True, what is the distribution of the prediction?(vice versa)")
        fig_preddist = prediction_distribution_plot()
        row1right.pyplot(fig_preddist)

    with st.container(border=True):
        row2left, row2right = st.columns(2)

        row2left.subheader("Threshold Performance")
        row2left.write("What would be the model performance if we change the prediction threshold?")
        row2left.write("")
        fig_thresholdperf = threshold_performance_plot()
        row2left.pyplot(fig_thresholdperf)

        row2right.subheader("ROC AUC Curve")
        row2right.write("How good the model at differentiating True Positive and False Positive ")
        fig_rocauc = roc_auc_plot()
        row2right.pyplot(fig_rocauc)


    

