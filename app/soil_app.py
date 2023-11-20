import streamlit as st
import datetime
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Lucas Soil Dataset Analysis",
    layout='wide',
    initial_sidebar_state='auto',
)

st.title("Lucas Soil Dataset Analysis")
st.subheader("Exploring Soil Characteristics and Distribution")
st.write("This project aims to utilize the LUCAS soil dataset to analyze the relationship between soil chemistry and soil biodiversity. The project will focus on understanding how the distribution of different soil nutrients and properties, such as pH, electrical conductivity (EC), organic carbon (OC), and inorganic nitrogen (N) affects the abundance and diversity of soil microorganisms. The project will also investigate the potential for soil chemistry to serve as an early warning system for the emergence of soil-borne pathogens, therefore making significant contributions to our understanding of soil health and the challenges facing agricultural soils. It will also provide valuable insights for policymakers and land managers who are working to improve soil health and protect the environment.")
st.write("For additional information view our video explaining the project at https://www.youtube.com/")

st.sidebar.header("Adjust Soil Data")
ph_val = st.sidebar.slider('pH', min_value=0, max_value=14)
ec_val = st.sidebar.slider('EC', min_value=0, max_value=10)
oc_val = st.sidebar.slider('OC', min_value=0, max_value=10)
car_val = st.sidebar.slider('CaCO3', min_value=0, max_value=10)
nit_val = st.sidebar.slider('N', min_value=0, max_value=10)
pho_val = st.sidebar.slider('P', min_value=0, max_value=10)
kal_val = st.sidebar.slider('K', min_value=0, max_value=10)
st.sidebar.header('Select Timeframe')
start_date = st.sidebar.date_input('Start Date', datetime.date(2017, 8, 19))
end_date = st.sidebar.date_input('End Date', datetime.date(2023, 11, 20))

countries = [
    'All', 'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Republic of Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden'
]

st.subheader('Select Region')
selected_options = st.multiselect('', countries)

if "All" in selected_options:
    selected_options = countries[1:] # selected_options keeps track of which countries are selected

df = pd.DataFrame({
    "col1": np.random.randn(1000) / 50 + 37.76,
    "col2": np.random.randn(1000) / 50 + -122.4,
    "col3": np.random.randn(1000) * 100,
    "col4": np.random.rand(1000, 4).tolist(),
})

st.map(df,
    latitude='col1',
    longitude='col2',
    size='col3',
    color='col4')