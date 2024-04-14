import pandas as pd # data science library o manipulate data
import numpy as np # mathematical library to manipulate arrays and matrices
import matplotlib.pyplot as plt # visualization ~
import urllib.request
import os
import re
from statsmodels.tsa.ar_model import AutoReg
#from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn import  metrics
import statsmodels.api as sm

def evaluating_models(test,predictions):
    MAE=metrics.mean_absolute_error(test,predictions) #Mean Absolute Error
    MBE=np.mean(test-predictions) #  Mean Bias Error
    MSE=metrics.mean_squared_error(test,predictions)  
    RMSE= np.sqrt(metrics.mean_squared_error(test,predictions))# Root Mean Squared Error  
    cvRMSE=RMSE/np.mean(test) #cv Root Mean Squared Error  
    NMBE=MBE/np.mean(test) #Normalized Mean Bias Error:
    return MAE, MBE,MSE,RMSE,cvRMSE,NMBE

############################################### CRIAR FOLDERS PARA GUARDAR AS IMAGENS ########################################################################
"""
code = "Portugal"
import requests
import json
response_API = requests.get(f'http://api.weatherapi.com/v1/{code}/last7days?key=CEJX4UTD2CUXMS3ZUQM3474D5&include=days')
data = response_API.text
print(data)
#parse_json = json.loads(data) &elements=temp,humidity,precip,windspeed,sealevelpressure,solarradiation'
"""

emissions_countries = r'emissions_countries' 
if not os.path.exists(emissions_countries):
    os.makedirs(emissions_countries)
    
weatherf = r'weatherf' 
if not os.path.exists(weatherf):
    os.makedirs(weatherf)
    
elect_prod = r'elect_prod' 
if not os.path.exists(elect_prod):
    os.makedirs(elect_prod)
    
flights = r'flights' 
if not os.path.exists(flights):
    os.makedirs(flights)

features_tests = r'features_tests' 
if not os.path.exists(features_tests):
    os.makedirs(features_tests)   
    
regressions = r'regressions' 
if not os.path.exists(regressions):
    os.makedirs(regressions)  

co2 = r'regressions/co2' 
if not os.path.exists(co2):
    os.makedirs(co2)

energy = r'regressions/energy' 
if not os.path.exists(energy):
    os.makedirs(energy)
    
forecast = r'forecast' 
if not os.path.exists(forecast):
    os.makedirs(forecast)
    
co2 = r'forecast/co2' 
if not os.path.exists(co2):
    os.makedirs(co2)

energy = r'forecast/energy' 
if not os.path.exists(energy):
    os.makedirs(energy)
    
csv_files = r'csv_files' 
if not os.path.exists(csv_files):
    os.makedirs(csv_files)
    
##############################################################################################################################################################

blue=["deepskyblue","skyblue","steelblue","dodgerblue","royalblue","blue"]

############################################################## TRATAMENTO DOS DADOS ##########################################################################


carbon_eu = pd.read_csv('carbon_eu.csv')  #dados das emissoes de carbono para todos os paises e os setores onde libertam

energy_prod = pd.read_csv('energy_prod.csv') #dados da produção de energia para todos os paises e os tipos de energia

carbon_eu=carbon_eu.interpolate(method="linear")

energy_prod=energy_prod.interpolate(method="linear")

date_range = pd.date_range(start='2021-01-01', end='2023-01-01', freq='D') #O intervalo de datas com as quais vamos trabalhar

countries=["Czech Republic","Cyprus","United Kingdom","Portugal","Spain","France","Germany","Belgium","Italy","Netherlands","Austria","Poland", "Luxembourg","Ireland","Finland","Sweden","Croatia","Slovenia",'Switzerland','Slovakia','Hungary',
           'Greece','Denmark','Romania','Latvia','Lithuania',"Estonia"] #Colocar mais paises à medida que conseguimos descarregar (é importante meter o resto é automático)

weather_cond=['Temperature [ºC]','Wind Speed [km/h]','Relative Humidity (%)','Pressure [mbar]','Solar Radiation [W/m^2]','Rain [mm/h]'] #Weather conditions que escolhemos

sectors_co2=list(set(carbon_eu["sector"])) #criar uma lista para os vários setores do CO2

sectors_en=list(set(energy_prod["sector"])) #criar uma lista para os varios tipos de energias

years=["2021","2022","2023"] #anos que vamos ver

############################## Tratamento dos dados em loop para os vários paises e a junção de todos os ficheiros excel ##########################

for country in countries:

    meteo = pd.read_csv(f'meteo_data/meteo_{country}.csv')
    
    meteo= meteo.interpolate(method="linear")
    
    meteo.index = date_range
    
    meteo.index.name="Date"
    

    meteo.rename(columns={'temp': 'Temperature [ºC]', 'windspeed': 'Wind Speed [km/h]',"humidity": 'Relative Humidity (%)',
                                'sealevelpressure': 'Pressure [mbar]','solarradiation': 'Solar Radiation [W/m^2]', 'precip': 'Rain [mm/h]'}, inplace=True)

    country_name = country
        
    df_country = carbon_eu[(carbon_eu['country'] == country_name)] 
    
    
    for sector in sectors_co2:
        print(sector)
        sector_list = df_country[(df_country['sector'] == sector)]
        sector_list = sector_list.drop(columns=['timestamp','sector'])
        sector_list.rename(columns={'value': f'{sector}', 'date': 'Date'}, inplace=True)
        globals()[f"{sector.replace(' ', '_')}"]= sector_list 

    df_country = pd.merge(Power, Ground_Transport, on=["Date","country"], how='inner')

    # Merge all types of energy 

    df_country  = pd.merge(df_country , International_Aviation, on=["Date","country"], how='inner')

    df_country  = pd.merge(df_country , Residential, on=["Date","country"], how='inner')

    df_country  = pd.merge(df_country , Industry, on=["Date","country"], how='inner')

    df_country  = pd.merge(df_country , Domestic_Aviation, on=["Date","country"], how='inner')

    print(df_country)
    
    df_country["Total CO2 [Tonne]"]= df_country["Power"]+df_country["International Aviation"]+df_country["Residential"]+df_country["Industry"]+df_country["Domestic Aviation"]

    df_country[['Day','Month','Year']] = df_country.Date.str.split("/",expand=True) 

    df_country = df_country.drop(columns=['Date'])

    df_country['Date'] = pd.to_datetime(df_country[['Day', 'Month', 'Year']])

    df_country = df_country.drop(columns=['Year', 'Month', 'Day'])

    df_country.set_index('Date', inplace=True)
    
    df_en = energy_prod[(energy_prod['country'] == country_name)] 
    
    for sector in sectors_en:
        sector_list = df_en[(df_en['sector'] == sector)]
        sector_list = sector_list.drop(columns=['timestamp','sector'])
        sector_list.rename(columns={'value': f'{sector}', 'date': 'Date'}, inplace=True)
        globals()[f"{sector.replace(' ', '_')}"]= sector_list 

    df_en = pd.merge(Other_sources, Gas, on=["Date","country"], how='inner')

    # Merge all types of energy 

    df_en  = pd.merge(df_en , Oil, on=["Date","country"], how='inner')

    df_en  = pd.merge(df_en , Coal, on=["Date","country"], how='inner')

    df_en  = pd.merge(df_en ,  Wind, on=["Date","country"], how='inner')

    df_en  = pd.merge(df_en ,Nuclear, on=["Date","country"], how='inner')
    
    df_en  = pd.merge(df_en ,Solar, on=["Date","country"], how='inner')
    
    df_en  = pd.merge(df_en ,Hydroelectricity, on=["Date","country"], how='inner')
    
    print(country)
    
    df_en["Total Renewable [GWh]"]= df_en["Wind"]+ df_en["Solar"] + df_en["Hydroelectricity"]

    df_en["Total Non-Renewable [GWh]"]= df_en["Other sources"]+df_en["Gas"]+df_en["Oil"]+df_en["Coal"]+ df_en["Nuclear"]
    
    df_en["Total Electricity [GWh]"]= df_en["Other sources"]+df_en["Gas"]+df_en["Oil"]+df_en["Coal"]+df_en["Wind"]+ df_en["Nuclear"] + df_en["Solar"] + df_en["Hydroelectricity"]
    
    df_en[['Day','Month','Year']] = df_en.Date.str.split("/",expand=True) 
    

    df_en = df_en.drop(columns=['Date'])

    df_en['Date'] = pd.to_datetime(df_en[['Day', 'Month', 'Year']])

    df_en = df_en.drop(columns=['Year', 'Month', 'Day'])

    df_en.set_index('Date', inplace=True)
    
    df_country  = pd.merge(df_country , meteo, on=["Date"], how='inner')
    
    df_country  = pd.merge(df_country , df_en, on=["Date","country"], how='inner')
    
    df_country=df_country.interpolate(method="linear")
    
    df_list=[]
    
    for year in years:
        df = pd.read_excel(f"flights_data/{year}flight.xlsx", sheet_name="Data")
        df = df[(df['Entity'] == country)] 
        df_list.append(df)
    
    flights= pd.concat(df_list)
    
    flights['Year'] = flights['Day'].dt.year
    flights['Month'] = flights['Day'].dt.month
    flights['Day'] = flights['Day'].dt.day

    flights['Date'] = pd.to_datetime(flights[['Year', 'Month', 'Day']])

    flights = flights.drop(columns=['Year','Week', 'Month', 'Day','Flights (7-day moving average)','Day 2019', 'Flights 2019 (Reference)',	'% vs 2019 (Daily)','% vs 2019 (7-day Moving Average)',	'Day Previous Year',	'Flights Previous Year'])

    flights.set_index('Date', inplace=True)
    
    flights.rename(columns={'Entity':'country'}, inplace=True)

    
    df_country  = pd.merge(df_country , flights, on=["Date","country"], how='inner')
    
    globals()[f"{country.replace(' ', '_')}"]= df_country
    
    globals()[f"{country.replace(' ', '_')}"]=globals()[f"{country.replace(' ', '_')}"].interpolate(method="linear")
    
    if "datetime" in globals()[f"{country.replace(' ', '_')}"].columns:
        globals()[f"{country.replace(' ', '_')}"] = globals()[f"{country.replace(' ', '_')}"].drop(columns=['datetime'])
    

    globals()[f"{country.replace(' ', '_')}"].to_csv(f"csv_files/{country}.csv")

df_total_country=globals()[f"{countries[0].replace(' ', '_')}"]

df_total_country=df_total_country[(df_total_country.index >= "2021-01-01") & 
                                     (df_total_country.index <= "2022-12-31")]
for country in countries[1:]:
    df_total_a_country=globals()[f"{country.replace(' ', '_')}"]
    df_total_a_country=df_total_a_country[(df_total_a_country.index >= "2021-01-01") & 
                                     (df_total_a_country.index <= "2022-12-31")]
    df_total_country=pd.concat([df_total_country,df_total_a_country])
    df_total_country.dropna(inplace=True)  
    

df_total_country.rename(columns={
    "Power": "Power [Tonne]",
    "Ground Transport": "Ground Transport [Tonne]" ,
    "International Aviation": "International Aviation [Tonne]",
    "Residential": "Residential [Tonne]",
    "Industry": "Industry [Tonne]",
    "Domestic Aviation" : "Domestic Aviation [Tonne]",
    "Nuclear":  "Nuclear [GWh]",
    "Gas" : "Gas [GWh]" ,
    "Oil" : "Oil [GWh]",
    "Coal": "Coal [GWh]",
    "Wind" : "Wind [GWh]",
    "Solar" : "Solar [Gwh]",
    "Hydroelectricity" : "Hydroelectricity [GWh]",
    "Other sources" : "Other sources [GWh]"
})
df_total_country.to_csv(f"csv_files/Allcountries.csv")
    
