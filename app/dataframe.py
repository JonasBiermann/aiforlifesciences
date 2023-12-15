import pandas as pd
import matplotlib as plt
import matplotlib.colors as mcolors
import pathlib
import os

path = os.path.dirname(__file__)
lucas_data_path = path+'/artifacts/lucas_soil_2018.csv'
shannon = path+'/artifacts/shannon.csv'

df1 = pd.read_csv(lucas_data_path)
df2 = pd.read_csv(shannon)
df = pd.merge(df1, df2, on='POINTID', how='outer', suffixes=('', '_y'))
df.drop(df.filter(regex='_y$').columns, axis=1, inplace=True)

predictions = df['predict']
color_map = plt.cm.get_cmap('RdYlGn')
norm = mcolors.Normalize(vmin=predictions.min(), vmax=predictions.max())
colors = [mcolors.to_hex(color_map(norm(value))) for value in predictions]
df['color'] = colors

df['CaCO3'] = df['CaCO3'].replace("<  LOD",-1).astype(float)
df['OC'] = df['OC'].replace("< LOD",-1).replace("<0.0",0).astype(float)

df['P']=df['P'].replace("< LOD",-1).replace("<0.0",0).astype(float)
df['N']=df['N'].replace("< LOD",-1).replace("<0.0",0).astype(float)
df['K']=df['K'].replace("< LOD",-1).replace("<0.0",0).astype(float)

df['SURVEY_DATE'] = pd.to_datetime(df['SURVEY_DATE'],format="%d-%m-%y")

country_codes = pd.read_html('https://en.wikipedia.org/wiki/Nomenclature_of_Territorial_Units_for_Statistics')[0]['Countries']
country_codes.columns = ['country','code']
country_mapping = dict(zip(country_codes['code'],country_codes['country']))
country_mapping['EL'] = 'Greece'
country_mapping['UK'] = 'United Kingdom'

df['COUNTRY'] = df['NUTS_0'].map(country_mapping)

area_categories = {
    'Forestry': 'Natural and Semi-natural Areas',
    'Agriculture (excluding fallow land and kitchen gardens)': 'Natural and Semi-natural Areas',
    'Road transport': 'Infrastructure and Utilities',
    'Semi-natural and natural areas not in use': 'Natural and Semi-natural Areas',
    'Electricity, gas and thermal power distribution': 'Infrastructure and Utilities',
    'Construction': 'Industrial and Commercial Areas',
    'Residential': 'Residential and Abandoned Areas',
    'Other abandoned areas': 'Residential and Abandoned Areas',
    'Energy production': 'Industrial and Commercial Areas',
    'Fallow land': 'Natural and Semi-natural Areas',
    'Amenities, museum, leisure (e.g. parks, botanical gardens)': 'Natural and Semi-natural Areas',
    'Kitchen gardens': 'Natural and Semi-natural Areas',
    'Railway transport': 'Infrastructure and Utilities',
    'Financial, professional and information services': 'Services and Miscellaneous',
    'Protection infrastructures': 'Infrastructure and Utilities',
    'Commerce': 'Industrial and Commercial Areas',
    'Mining and quarrying': 'Industrial and Commercial Areas',
    'Sport': 'Services and Miscellaneous',
    'Community services': 'Services and Miscellaneous',
    'Water supply and treatment': 'Infrastructure and Utilities',
    'Other primary production': 'Services and Miscellaneous',
    'Abandoned residential areas': 'Residential and Abandoned Areas',
    'Logistics and storage': 'Industrial and Commercial Areas',
    'Abandoned industrial areas': 'Residential and Abandoned Areas',
    'Water transport': 'Services and Miscellaneous',
    'Abandoned transport areas': 'Residential and Abandoned Areas',
}

df['category'] = df['LU1_Desc'].map(area_categories)