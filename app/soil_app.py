import streamlit as st
import datetime
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Lucas Soil Dataset Analysis",
    layout='wide',
    initial_sidebar_state='auto',
)

df = pd.read_csv('/Volumes/Data 1/dev/aiforlifesciences/app/LUCAS SOIL 2018.csv')
df['CaCO3'] = df['CaCO3'].replace("<  LOD",-1).astype(float)
df['OC'] = df['OC'].replace("< LOD",-1).replace("<0.0",0).astype(float)

df['P']=df['P'].replace("< LOD",-1).replace("<0.0",0).astype(float)
df['N']=df['N'].replace("< LOD",-1).replace("<0.0",0).astype(float)
df['K']=df['K'].replace("< LOD",-1).replace("<0.0",0).astype(float)

df['SURVEY_DATE'] = pd.to_datetime(df['SURVEY_DATE'],format="%d-%m-%y")

st.title("Lucas Soil Dataset Analysis")
st.subheader("Exploring Soil Characteristics and Distribution")
st.write("This project aims to utilize the LUCAS soil dataset to analyze the relationship between soil chemistry and soil biodiversity. The project will focus on understanding how the distribution of different soil nutrients and properties, such as pH, electrical conductivity (EC), organic carbon (OC), and inorganic nitrogen (N) affects the abundance and diversity of soil microorganisms. The project will also investigate the potential for soil chemistry to serve as an early warning system for the emergence of soil-borne pathogens, therefore making significant contributions to our understanding of soil health and the challenges facing agricultural soils. It will also provide valuable insights for policymakers and land managers who are working to improve soil health and protect the environment.")
st.write("For additional information view our video explaining the project at https://www.youtube.com/")

st.sidebar.header("Adjust Soil Data")
ph_val = st.sidebar.slider('pH', min_value=0.0, max_value=14.0, value=(min(df['pH_CaCl2']), max(df['pH_CaCl2'])))
ec_val = st.sidebar.slider('EC', min_value=min(df['EC']), max_value=max(df['EC']), value=(min(df['EC']), max(df['EC'])))
oc_val = st.sidebar.slider('OC', min_value=float(min(df['OC'])), max_value=float(max(df['OC'])), value=(float(min(df['OC'])), float(max(df['OC']))))
car_val = st.sidebar.slider('CaCO3', min_value=min(df['CaCO3']), max_value=max(df['CaCO3']), value=(min(df['CaCO3']), max(df['CaCO3'])))
pho_val = st.sidebar.slider('P', min_value=min(df['P']), max_value=max(df['P']), value=(min(df['P']), max(df['P'])))
nit_val = st.sidebar.slider('N', min_value=min(df['N']), max_value=max(df['N']), value=(min(df['N']), max(df['N'])))
kal_val = st.sidebar.slider('K', min_value=min(df['K']), max_value=max(df['K']), value=(min(df['K']), max(df['K'])))
st.sidebar.header('Select Timeframe')
start_date = pd.to_datetime(st.sidebar.date_input('Start Date', datetime.date(2017, 8, 19), format="DD/MM/YYYY"))
end_date = pd.to_datetime(st.sidebar.date_input('End Date', datetime.date(2023, 11, 20), format="DD/MM/YYYY"))

countries = [
    'All', 'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Republic of Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden'
]

st.subheader('Select Region')
selected_options = st.multiselect('', countries)

if "All" in selected_options:
    selected_options = countries[1:] # selected_options keeps track of which countries are selected

df = df[
    (df['SURVEY_DATE']>=start_date) & 
    (df['SURVEY_DATE']<=end_date) &
    (df['pH_CaCl2'] >= ph_val[0]) & 
    (df['pH_CaCl2'] <= ph_val[1]) &
    (df['EC'] >= ec_val[0]) &
    (df['EC'] <= ec_val[1]) &
    (df['OC'] >= oc_val[0]) &
    (df['OC'] <= oc_val[1]) &
    (df['CaCO3'] >= car_val[0]) &
    (df['CaCO3'] <= car_val[1]) &
    (df['N'] >= nit_val[0]) &
    (df['N'] <= nit_val[1]) &
    (df['P'] >= pho_val[0]) &
    (df['P'] <= pho_val[1]) &
    (df['K'] >= kal_val[0]) &
    (df['K'] <= kal_val[1])
]

st.map(df, latitude='TH_LAT', longitude="TH_LONG", use_container_width=True)