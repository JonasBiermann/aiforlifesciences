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
    threshold_performance_plot,
    shap_pdp_plot,
    roc_auc_plot,
    total_error_plot,
    error_per_fold_plot,
    residual_per_feature_plot)

shap.initjs()

path = os.path.dirname(__file__)

st.set_page_config(page_title="Machine Learning and Explainability", page_icon="🤖", layout='wide')

st.markdown("# Predicting Soil Biodiversity with Machine Learning")

st.subheader("Description")
st.write("In our collaborative effort, we developed a robust machine learning model aimed at predicting soil biodiversity based on key soil characteristics. Leveraging the SHAP (SHapley Additive exPlanations) algorithm and classification techniques, our model draws insights from the LUCAS dataset, a comprehensive soil database.")
st.write("If you're interested about the detalis of this model just click Learn More!")
with st.expander('Learn More'):
    st.subheader("Data Selection")
    st.write("Our model is trained on the LUCAS dataset, which provides a rich source of soil information. The dataset includes crucial parameters such as pH, nitrogen (N), phosphorus (P), potassium (K), organic carbon (OC), and electrical conductivity (EC).")

    st.subheader("Feature Exploration")
    st.write("We meticulously explored the relationships between soil biodiversity and the selected features. pH levels, nutrient content (N, P, K), organic carbon, and electrical conductivity were identified as critical factors influencing soil biodiversity.")


    st.subheader("Classification")
    st.write("Our machine learning model employs a classification approach to predict soil biodiversity. By leveraging patterns within the LUCAS dataset, the model has learned to categorize soil samples into distinct biodiversity classes based on the specified features.")

    st.subheader("Validation and Fine-Tuning")
    st.write("We rigorously validated and fine-tuned the model to ensure its accuracy and generalizability. This involved splitting the dataset into training and testing sets, optimizing hyperparameters, and employing techniques to mitigate overfitting.")

    st.subheader("Results and Implications")
    st.write("The resulting model provides a reliable tool for predicting soil biodiversity, offering valuable insights for land management and conservation efforts. By understanding the influence of pH, nutrient levels, and other factors, our model contributes to a deeper understanding of soil ecosystems and supports informed decision-making in agriculture and environmental science. You can try it out yourself on the page Soil Data Check!")


model = cb.CatBoostRegressor(loss_function='RMSE')
catboost_model_path = path+'/../artifacts/CatBoostShannon.cb'

model.load_model(catboost_model_path)
explainer = shap.TreeExplainer( # I make a mistake here, this should be loaded instead of calculate on the fly like this
    model
)

shap_values_path = path+'/../artifacts/shap_valuesCatBoostShannon.jb'
model_performance_path = path+'/../artifacts/model_performance.csv'

shap_values = joblib.load(shap_values_path)
test_df = pd.read_csv(model_performance_path)

list_X = ['pH_CaCl2','pH_H2O','CaCO3','EC','OC','P','N','K','LC0_Desc','LC1_Desc','LU1_Desc']
list_X_cat = ['LC0_Desc','LC1_Desc','LU1_Desc']
list_X_num = [x for x in list_X if x not in list_X_cat]
df_shap = pd.DataFrame(shap_values.values,columns=list_X)
df_shap.reset_index(inplace=True)

df_raw = pd.DataFrame(shap_values.data,columns=list_X)
with st.expander('Model Performance Evaluation'):
    st.header("Model Performance")

    kpi1, kpi2,kpi3,kpi4 = st.columns(4)
    from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error
    num_kpi1 = mean_squared_error(test_df['target'],test_df['pred'],squared=False)
    num_kpi2 = mean_absolute_error(test_df['target'],test_df['pred'])
    num_kpi3 = r2_score(test_df['target'],test_df['pred'])
    num_kpi4 = mean_absolute_percentage_error(test_df['target'],test_df['pred'])

    kpi1.metric("RMSE",str(round(num_kpi1,3)))
    kpi2.metric("MAE",str(round(num_kpi2,3)))
    kpi3.metric("R-Square",str(round(num_kpi3,3)))
    kpi4.metric("MAPE",str(round(num_kpi4*100,2))+"%")

    with st.container(border=True):
        row1left, row1right = st.columns(2)

        row1left.subheader("Distribution Error plot")
        row1left.write("From the 10 fold validation, what is the distribution of the error ? ")
        row1left.write("")
        fig_total_error = total_error_plot(test_df)
        row1left.pyplot(fig_total_error)

        row1right.subheader("Fold-wise error")
        row1right.write("What is the distribution of the error in the 10 fold cross validation ? ")
        fig_error_per_fold = error_per_fold_plot(test_df)
        row1right.pyplot(fig_error_per_fold)

    with st.container(border=True):
        st.subheader("Error residuals across different predictors")
        st.write("Help to identify any biases or heteroskedasticity in the model")
        row2left, row2right = st.columns(2)
        tabs = st.tabs(list_X)
        for tab,feature_name in zip(tabs,list_X) :
            fig_residual = residual_per_feature_plot(test_df,feature_name,list_X_cat)
            tab.pyplot(fig_residual)

with st.expander('Model Analysis - SHAP'):
    st.header("SHAP Analysis")
    image_path = os.path.join(script_dir, "..", "artifacts", "shap_logo.png")
    st.image(image_path)
    st.subheader("Global Analysis")
    st.write("""To unravel the intricate connections between features and biodiversity, we employed the SHAP algorithm. 
    SHAP values offer insightful explanations for individual predictions, shedding light on the contribution of each feature to the model's output. 
    This interpretability is crucial for understanding the model's decision-making process.""")

    
    with st.container(border=True):
        fig_beeswarm,_ = beeswarm_plot(df_raw,df_shap,list_X_num,list_X_cat)
        st.pyplot(fig_beeswarm)

    st.subheader("Partial Dependence Plot")
    tabs = st.tabs(shap_values.feature_names)
    for tab,feature_name in zip(tabs,shap_values.feature_names) :
        fig_pdp = shap_pdp_plot(shap_values,feature_name,list_X_cat)
        tab.pyplot(fig_pdp)


        


    

