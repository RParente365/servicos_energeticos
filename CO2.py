import pandas as pd # data science library o manipulate data
import numpy as np # mathematical library to manipulate arrays and matrices
import matplotlib.pyplot as plt # visualization ~
import urllib.request
import os
import re

############################################### CRIAR FOLDERS PARA GUARDAR AS IMAGENS ########################################################################


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
    
##############################################################################################################################################################

blue=["deepskyblue","skyblue","steelblue","dodgerblue","royalblue","blue"]

############################################################## TRATAMENTO DOS DADOS ##########################################################################


carbon_eu = pd.read_csv('carbon_eu.csv')  #dados das emissoes de carbono para todos os paises e os setores onde libertam

energy_prod = pd.read_csv('energy_prod.csv') #dados da produção de energia para todos os paises e os tipos de energia

carbon_eu=carbon_eu.dropna() 

energy_prod=energy_prod.dropna() 

date_range = pd.date_range(start='2021-01-01', end='2023-01-01', freq='D') #O intervalo de datas com as quais vamos trabalhar

countries=["Portugal","Spain","France","Germany"] #Colocar mais paises à medida que conseguimos descarregar (é importante meter o resto é automático)

weather_cond=['Temperature [ºC]','Wind Speed [km/h]','Relative Humidity (%)','Pressure [mbar]','Solar Radiation [W/m^2]','Rain [mm/h]'] #Weather conditions que escolhemos

sectors_co2=list(set(carbon_eu["sector"])) #criar uma lista para os vários setores do CO2

sectors_en=list(set(energy_prod["sector"])) #criar uma lista para os varios tipos de energias

years=["2021","2022","2023"] #anos que vamos ver

############################## Tratamento dos dados em loop para os vários paises e a junção de todos os ficheiros excel ##########################

for country in countries:
    
    meteo = pd.read_csv(f'meteo_{country}.csv')
    
    meteo.index = date_range
    
    meteo.index.name="Date"
    

    meteo.rename(columns={'temp': 'Temperature [ºC]', 'windspeed': 'Wind Speed [km/h]',"humidity": 'Relative Humidity (%)',
                                'sealevelpressure': 'Pressure [mbar]','solarradiation': 'Solar Radiation [W/m^2]', 'precip': 'Rain [mm/h]'}, inplace=True)


    
    df_country = carbon_eu[(carbon_eu['country'] == country)] 
    
    for sector in sectors_co2:
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

    
    df_country["Total CO2 [Tonne]"]= df_country["Power"]+df_country["International Aviation"]+df_country["Residential"]+df_country["Industry"]+df_country["Domestic Aviation"]

    df_country[['Day','Month','Year']] = df_country.Date.str.split("/",expand=True) 

    df_country = df_country.drop(columns=['Date'])

    df_country['Date'] = pd.to_datetime(df_country[['Day', 'Month', 'Year']])

    df_country = df_country.drop(columns=['Year', 'Month', 'Day'])

    df_country.set_index('Date', inplace=True)
    
    df_en = energy_prod[(energy_prod['country'] == country)] 
    
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
    
    df_country=df_country.dropna() 
    
    df_list=[]
    
    for year in years:
        df = pd.read_excel(f"{year}flight.xlsx", sheet_name="Data")
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
    
    globals()[f"{country.replace(' ', '_')}"]=globals()[f"{country.replace(' ', '_')}"].dropna() 
    
    if "datetime" in globals()[f"{country.replace(' ', '_')}"].columns:
        globals()[f"{country.replace(' ', '_')}"] = globals()[f"{country.replace(' ', '_')}"].drop(columns=['datetime'])
        

#Uma coisa importante daqui é que o ficheiro final é o globals()[f"{country.replace(' ', '_')}"] isto porque a partida vamos sempre fazer um loop para os countries
# e não nos necessitamos de preocupar em dar um nome um a um. Este é o nome do nosso dataframe final

################################################# Plots só para analisar se a data está toda mais ou menos em ordem ################################################

for country in countries:
    i=0
    
    plt.figure(figsize=(12,6))   
    for sector in sectors_co2:
        plt.plot(globals()[f"{country.replace(' ', '_')}"][f'{sector}'], 
                 label=sector, color=blue[i])
        i=i+1
        plt.legend()

    plt.xlabel("Date")
    plt.ylabel("CO^2 Emissions in Tonne")
    plt.title(f'{country} emissions for each sector')
    plt.savefig(f"emissions_countries/{country}_emissions.png") 
    plt.clf()
    
    plt.figure(figsize=(12,6))
    i=0
    for sector in sectors_en:
        plt.plot(globals()[f"{country.replace(' ', '_')}"][f'{sector}'], 
                 label=sector)
        i=i+1
        plt.legend()

    plt.xlabel("Date")
    plt.ylabel("Energy Production in GWh")
    plt.title(f'{country} Energy Production for each sector')
    plt.savefig(f"elect_prod/{country}_elect_prod.png") 
    plt.clf()
    
    plt.figure(figsize=(12,6))   
    
    for weather in weather_cond:
        weather_name = re.split(r'\s|\[', weather)[0]
        plt.plot(globals()[f"{country.replace(' ', '_')}"][f'{weather}'], 
                 label=weather)
        i=i+1
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel(f"{weather}")
        plt.title(f'{weather} for {country}')
        plt.savefig(f"weatherf/{country}_{weather_name}.png") 
        plt.clf()
 
 
plt.figure(figsize=(12,6))   

for country in countries:
    
    plt.plot(globals()[f"{country.replace(' ', '_')}"]['Total CO2 [Tonne]'], 
                label=country)   

plt.legend(loc="upper center",bbox_to_anchor=(0.5, -0.2),ncol=8)
plt.subplots_adjust(bottom=0.3)
plt.xlabel("Date")
plt.ylabel("CO^2 Emissions in Tonne")
plt.title(f'Total emissions for each country')
plt.savefig(f"emissions_countries/Total_emissions.png") 
plt.clf()


for country in countries:
    
    plt.plot(globals()[f"{country.replace(' ', '_')}"]['Total Electricity [GWh]'], 
                label=country)
    
plt.legend(loc="upper center",bbox_to_anchor=(0.5, -0.2),ncol=8)
plt.subplots_adjust(bottom=0.3)
plt.xlabel("Date")
plt.ylabel("Total Energy Producted in GWh")
plt.title(f'Total Energy Production for each country')
plt.savefig(f"elect_prod/Total_prod.png") 
plt.clf()

for country in countries:
    
    plt.plot(globals()[f"{country.replace(' ', '_')}"]['Flights'], 
                label=country)
    
plt.legend(loc="upper center",bbox_to_anchor=(0.5, -0.2),ncol=8)
plt.subplots_adjust(bottom=0.3)
plt.xlabel("Date")
plt.ylabel("Number of Flights per day")
plt.title(f'Number of flights for each country')
plt.savefig(f"flights/flights.png") 
plt.clf()


for country in countries:
    
    plt.plot(globals()[f"{country.replace(' ', '_')}"]['Total Renewable [GWh]'], 
                label= "Renewable Sources (Wind, Solar, Hydro)")
    plt.plot(globals()[f"{country.replace(' ', '_')}"]['Total Non-Renewable [GWh]'], 
                label= "Non-Renewable Sources (Oil, Coal, Gas, Nuclear, Other Sources)")
    
    plt.legend(loc="upper center",bbox_to_anchor=(0.5, -0.2),ncol=8)
    plt.subplots_adjust(bottom=0.3)
    plt.xlabel("Date")
    plt.ylabel("Energy Producted in GWh")
    plt.title(f'Renewable vs Non-Renewable energy production {country}')
    plt.savefig(f"elect_prod/Total_prod_{country}.png") 
    plt.clf()

################################################ Começar a fazer as features selections ########################################################################################
 
from sklearn.feature_selection import SelectKBest # selection method
from sklearn.feature_selection import mutual_info_regression,f_regression

for country in countries:
     
    #só estou a meter como outputs o CO2 emission e o Renewable energy (isoladamente) e aqui a criar a média dos ultimos 3 valores e o valor anterior
    
    globals()[f"{country.replace(' ', '_')}"]['CO2 Emission-1 [kW]']=globals()[f"{country.replace(' ', '_')}"]['Total CO2 [Tonne]'].shift(1) # Previous hour consumption

    globals()[f"{country.replace(' ', '_')}"]["3 Last CO2 Mean"] = globals()[f"{country.replace(' ', '_')}"]['Total CO2 [Tonne]'].rolling(window=4).apply(lambda x: x[-4:-1].mean(), raw=True) #mean of the last three consumption values for each index

    globals()[f"{country.replace(' ', '_')}"]['Ren Energy-1 [kW]']=globals()[f"{country.replace(' ', '_')}"]['Total Renewable [GWh]'].shift(1) # Previous hour consumption

    globals()[f"{country.replace(' ', '_')}"]["3 Last Ren Energy Mean"] = globals()[f"{country.replace(' ', '_')}"]['Total Renewable [GWh]'].rolling(window=4).apply(lambda x: x[-4:-1].mean(), raw=True) #mean of the last three consumption values for each index
    
    globals()[f"{country.replace(' ', '_')}"]=globals()[f"{country.replace(' ', '_')}"].dropna() 
    
    # Fica aqui uma lista só para ser mais fácil identificar o indice de cada variável
    
    """
    0 : country', 
    1 :'Power', 
    2: 'Ground Transport', 
    3:'International Aviation',
    4:   'Residential', 
    5:   'Industry', 
    6:   'Domestic Aviation', 
    7:   'Total CO2 [Tonne]',
    8:   'Temperature [ºC]', 
    9:   'Relative Humidity (%)', 
    10:   'Rain [mm/h]',
    11:   'Wind Speed [km/h]', 
    12:   'Pressure [mbar]', 
    13:   'Solar Radiation [W/m^2]',
    14:   'Other sources', 
    15:   'Gas', 
    16:   'Oil', 
    17:   'Coal', 
    18:   'Wind', 
    19:   'Nuclear', 
    20:   'Solar',
    21:   'Hydroelectricity', 
    22:   'Total Renewable [GWh]',
    23:   'Total Non-Renewable [GWh]', 
    24:   'Total Electricity [GWh]', 
    25:   'Flights',
    26:   'CO2 Emission-1 [kW]', 
    27:   '3 Last CO2 Mean', 
    28:   'Ren Energy-1 [kW]',
    29   '3 Last Ren Energy Mean'],
    
    """
    
    ############################# Para o CO2 emissions ##################################################################################################
    
    Z= globals()[f"{country.replace(' ', '_')}"].values

    Y_co=Z[:,7]
    
    print(globals()[f"{country.replace(' ', '_')}"].columns)
    
    X_co=Z[:,[8,9,10,11,12,13,14,15,16,17,18,19,20,21,25,26,27]] 

    features=SelectKBest(k=2,score_func=mutual_info_regression)

    fit=features.fit(X_co,Y_co) #calculates the scores using the score_function f_regression of the features
    features_results=fit.transform(X_co)

    plt.figure(figsize=(17,6))
    
    features = globals()[f"{country.replace(' ', '_')}"].columns.tolist()[8:22] + globals()[f"{country.replace(' ', '_')}"].columns.tolist()[25:28]
    
    plt.bar(features, fit.scores_, color="red")

    plt.xlabel("k")
    plt.ylabel("Fit Score")
    plt.title(f'{country}')
    plt.xticks(rotation=90) 
    plt.subplots_adjust(bottom=0.27)
    plt.savefig(f"features_tests/fitscore_CO2_{country}.png") 
    plt.clf()
    
    high_feat_temp=dict(zip(features, fit.scores_))

    high_feat_temp = dict(sorted(high_feat_temp.items(), key=lambda item: item[1], reverse=True))
    
    print(f"Highest features scores for CO2 as Output with kBest for {country}:\n")
    for i, (column, score) in enumerate(high_feat_temp.items()):
        if i < 7:
            print(f"{column}: {score}")
        else:
            break
    print("\n")
    
    ############################################ Para a Ren Energy ##################################################################################
    
    Y_en=Z[:,22]
    
    X_en=Z[:,[1,2,3,4,5,6,8,9,10,11,12,13,25,28,29]] 

    features=SelectKBest(k=2,score_func=mutual_info_regression)

    fit=features.fit(X_en,Y_en) #calculates the scores using the score_function f_regression of the features
    features_results=fit.transform(X_en)

    plt.figure(figsize=(17,6))
    
    features = globals()[f"{country.replace(' ', '_')}"].columns.tolist()[1:7] + globals()[f"{country.replace(' ', '_')}"].columns.tolist()[8:14] + [globals()[f"{country.replace(' ', '_')}"].columns.tolist()[25]] + globals()[f"{country.replace(' ', '_')}"].columns.tolist()[28:]
    
    plt.bar(features, fit.scores_, color="lime")

    plt.xlabel("k")
    plt.ylabel("Fit Score")
    plt.title(f'{country}')
    plt.xticks(rotation=90) 
    plt.subplots_adjust(bottom=0.27)
    plt.savefig(f"features_tests/fitscore_Renewable_{country}.png") 
    plt.clf()
    
    high_feat_temp=dict(zip(features, fit.scores_))

    high_feat_temp = dict(sorted(high_feat_temp.items(), key=lambda item: item[1], reverse=True))
    
    print(f"Highest features scores for Renewable Energy as Output with kBest for {country}:\n")
    for i, (column, score) in enumerate(high_feat_temp.items()):
        if i < 7:
            print(f"{column}: {score}")
        else:
            break
    print("\n")
    




