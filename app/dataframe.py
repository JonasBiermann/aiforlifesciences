import pandas as pd
import numpy as np

df = pd.read_csv('/workspaces/aiforlifesciences/app/LUCAS SOIL 2018.csv')
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