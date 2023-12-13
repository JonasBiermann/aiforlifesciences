import streamlit as st
from dataframe import *

st.set_page_config(
    page_title="Lucas Soil Dataset Analysis",
    layout='wide',
    initial_sidebar_state='auto',
)

st.title("Lucas Soil Dataset Analysis")
st.subheader("Exploring Soil Characteristics and Distribution")
st.write("This project aims to utilize the LUCAS soil dataset to analyze the relationship between soil chemistry and soil biodiversity. The project will focus on understanding how the distribution of different soil nutrients and properties, such as pH, electrical conductivity (EC), organic carbon (OC), and inorganic nitrogen (N) affects the abundance and diversity of soil microorganisms. The project will also investigate the potential for soil chemistry to serve as an early warning system for the emergence of soil-borne pathogens, therefore making significant contributions to our understanding of soil health and the challenges facing agricultural soils. It will also provide valuable insights for policymakers and land managers who are working to improve soil health and protect the environment.")
st.write("For additional information view our video explaining the project at https://www.youtube.com/")

countries = [
    'All', 'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Republic of Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden'
]

# for this landuse the clustering of the land usages would be cool
# also it makes more sense to change the markers color depending on the biodiversity
# of a given place therefore enabling the user to filter by land use
# and then being able maybe to give a summary which types of land have the best/worst biodiversity

land_use = [
    'Natural and Semi-natural Areas',
    'Infrastructure and Utilities',
    'Industrial and Commercial Areas',
    'Residential and Abandoned Areas',
    'Services and Miscellaneous',
]

col1, col2 = st.columns([3, 2])
with col1:
    st.subheader('Select Region')
    selected_countries = st.multiselect('', countries)
with col2:
    st.subheader('Select Land Use')
    category = st.selectbox('', land_use)

if "All" in selected_countries:
    selected_countries = countries[1:] # selected_options keeps track of which countries are selected

st.write(category = list(df['category'].unique())[0])
df_filter = df[df['category'] == category]

if 'All' not in selected_countries:
    df_map = df_filter[(df_filter['COUNTRY'].isin(selected_countries))]

df_length = len(df_map)
# df_map.shape
if df_length > 0:
    st.map(df_map, latitude='TH_LAT', longitude="TH_LONG", use_container_width=True, color='color')
    st.write('The map visually represents soil samples, allowing users to filter by both land use and location parameters. The coloration of the data points on the map corresponds to the biodiversity level of each soil sample. Higher biodiversity is depicted in shades of green, creating a spectrum where the darkest green represents the highest biodiversity. Conversely, lower biodiversity is represented by shades of red, with the deepest red indicating the lowest biodiversity. This color-coded scheme enables users to quickly identify and analyze the biodiversity variations across different soil samples, making it an intuitive and informative visualization tool for assessing ecological richness in specific land use and location contexts.')
else:
    if selected_countries and category:
        st.warning('The configuration you selected yielded no results. Choose a different country or land usage category to continue!', icon="⚠️")