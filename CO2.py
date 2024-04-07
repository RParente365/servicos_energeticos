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
import plotly.express as px

def evaluating_models(test,predictions):
    MAE=metrics.mean_absolute_error(test,predictions) #Mean Absolute Error
    MBE=np.mean(test-predictions) #  Mean Bias Error
    MSE=metrics.mean_squared_error(test,predictions)  
    RMSE= np.sqrt(metrics.mean_squared_error(test,predictions))# Root Mean Squared Error  
    cvRMSE=RMSE/np.mean(test) #cv Root Mean Squared Error  
    NMBE=MBE/np.mean(test) #Normalized Mean Bias Error:
    return MAE, MBE,MSE,RMSE,cvRMSE,NMBE

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
    
##############################################################################################################################################################

blue=["deepskyblue","skyblue","steelblue","dodgerblue","royalblue","blue"]

############################################################## TRATAMENTO DOS DADOS ##########################################################################


carbon_eu = pd.read_csv('carbon_eu.csv')  #dados das emissoes de carbono para todos os paises e os setores onde libertam

energy_prod = pd.read_csv('energy_prod.csv') #dados da produção de energia para todos os paises e os tipos de energia

carbon_eu=carbon_eu.dropna() 

energy_prod=energy_prod.dropna() 

date_range = pd.date_range(start='2021-01-01', end='2023-01-01', freq='D') #O intervalo de datas com as quais vamos trabalhar

countries=["Portugal","Spain","France","Germany","Belgium","Italy"] #Colocar mais paises à medida que conseguimos descarregar (é importante meter o resto é automático)

weather_cond=['Temperature [ºC]','Wind Speed [km/h]','Relative Humidity (%)','Pressure [mbar]','Solar Radiation [W/m^2]','Rain [mm/h]'] #Weather conditions que escolhemos

sectors_co2=list(set(carbon_eu["sector"])) #criar uma lista para os vários setores do CO2

sectors_en=list(set(energy_prod["sector"])) #criar uma lista para os varios tipos de energias

years=["2021","2022","2023"] #anos que vamos ver

############################## Tratamento dos dados em loop para os vários paises e a junção de todos os ficheiros excel ##########################

for country in countries:
    
    meteo = pd.read_csv(f'meteo_data/meteo_{country}.csv')
    
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

high_feat_temp_co = {}
high_feat_temp_en= {}

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
    
    kbest_points = {feature: (len(high_feat_temp) - i) for i, (feature, _) in enumerate(high_feat_temp.items())}
    
    print(f"Highest features scores for CO2 as Output with kBest for {country}:\n")
    for i, (column, score) in enumerate(high_feat_temp.items()):
        if i < 9:
            print(f"{column}: {score}")
        else:
            break
    print("\n")
    
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor() 
    model.fit(X_co, Y_co)

    high_feat_temp=dict(zip(features, model.feature_importances_))

    high_feat_temp = dict(sorted(high_feat_temp.items(), key=lambda item: item[1], reverse=True))
    
    rf_points = {feature: (len(high_feat_temp) - i) for i, (feature, _) in enumerate(high_feat_temp.items())}

    plt.figure(figsize=(12,6))

    print("Highest features importances for CO2 as Output with RandomForestRegressor:\n")
    for i, (column, score) in enumerate(high_feat_temp.items()):
        if i < 9:
            print(f"{column}: {score}")
            plt.plot(globals()[f"{country.replace(' ', '_')}"][column], label=f"{column}")
        else:
            break
    print("\n")
    

    # Combine points from both methods
    total_points = {}
    
    for feature in high_feat_temp.keys():
        total_points[feature] = rf_points.get(feature, 0) + kbest_points.get(feature, 0)

    # Sort features based on total points
    sorted_features = sorted(total_points.items(), key=lambda x: x[1], reverse=True)

    # Select the top features
    high_feat_temp_co[f"high_feat_temp_{country}"] = sorted_features[:9]
    
    
    print("Best features for CO2 emission after both tests:")
    print(high_feat_temp_co[f"high_feat_temp_{country}"])
    
    plt.figure(figsize=(12,6))
    
    for (column,value) in high_feat_temp_co[f"high_feat_temp_{country}"]:
        plt.plot(globals()[f"{country.replace(' ', '_')}"][column], label=f"{column}")
    plt.plot(globals()[f"{country.replace(' ', '_')}"]["Total CO2 [Tonne]"], label=f"Total CO^2 Emission")
    
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Variables")
    plt.title(f'{country}')
    plt.subplots_adjust(bottom=0.27)
    plt.savefig(f"features_tests/best_feat_{country}.png") 
    plt.clf()
    

    
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

    kbest_points = {feature: (len(high_feat_temp) - i) for i, (feature, _) in enumerate(high_feat_temp.items())}
    
    print(f"Highest features scores for Renewable Energy as Output with kBest for {country}:\n")
    for i, (column, score) in enumerate(high_feat_temp.items()):
        if i < 9:
            print(f"{column}: {score}")
        else:
            break
    print("\n")
    
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor() 
    model.fit(X_en, Y_en)

    high_feat_temp=dict(zip(features, model.feature_importances_))

    high_feat_temp = dict(sorted(high_feat_temp.items(), key=lambda item: item[1], reverse=True))
    
    rf_points = {feature: (len(high_feat_temp) - i) for i, (feature, _) in enumerate(high_feat_temp.items())}

    plt.figure(figsize=(12,6))

    print("Highest features importances for Renewable energy as Output with RandomForestRegressor:\n")
    for i, (column, score) in enumerate(high_feat_temp.items()):
        if i < 9:
            print(f"{column}: {score}")
            plt.plot(globals()[f"{country.replace(' ', '_')}"][column], label=f"{column}")
        else:
            break
    print("\n")


    plt.plot(globals()[f"{country.replace(' ', '_')}"]["Total Renewable [GWh]"], label=f"Renewable Energy")
    plt.legend()

    # Combine points from both methods
    total_points = {}
    
    for feature in high_feat_temp.keys():
        total_points[feature] =  kbest_points.get(feature, 0) + rf_points.get(feature, 0) 

    # Sort features based on total points
    sorted_features = sorted(total_points.items(), key=lambda x: x[1], reverse=True)

    # Select the top features
    high_feat_temp_en[f"high_feat_temp_{country}"] = sorted_features[:9]
    
    print("Best features for Renewable Energy after both tests:")
    print(high_feat_temp_en[f"high_feat_temp_{country}"])
    


################################################## Random Forest Model ##########################################################################

for country in countries:

    options=[high_feat_temp_co[f"high_feat_temp_{country}"],high_feat_temp_en[f"high_feat_temp_{country}"] ]
    
    for option in options:
        
        df_forecast=pd.DataFrame()

        if option==high_feat_temp_co[f"high_feat_temp_{country}"]:   
            df_forecast["Total CO2 [Tonne]"]=globals()[f"{country.replace(' ', '_')}"]["Total CO2 [Tonne]"]
        elif option==high_feat_temp_en[f"high_feat_temp_{country}"]:
            df_forecast["Total Renewable [GWh]"]=globals()[f"{country.replace(' ', '_')}"]["Total Renewable [GWh]"]
        
        for feat in option:
            df_forecast[feat[0]]=(globals()[f"{country.replace(' ', '_')}"][f"{feat[0]}"])
            
        Z=df_forecast.values
        
        Y=Z[:, 0]
        #Identify input Y
        X=Z[:,[1,2,3,4,5,6]]  # Temperature,Solar Radiation, Power-1, average value of last three data values

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        parameters = {'bootstrap': True,
                    'min_samples_leaf': 4,
                    'n_estimators': 1030, 
                    'min_samples_split': 2,
                    'max_features': "sqrt",
                    'max_depth': 70,
                    'max_leaf_nodes': None}

        RF_model = RandomForestRegressor(**parameters)

        X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

        RF_model.fit(X_train, Y_train)

        y_pred_RF = RF_model.predict(X_test)

        fig = plt.figure(figsize=(16,5))

        plt.subplot(1, 2, 1)

        plt.plot(Y_test)
        plt.plot(y_pred_RF)
        plt.show()
        plt.subplot(1, 2, 2)
        plt.scatter(Y_test,y_pred_RF)
        plt.show()

        MAE_RF, MBE_RF,MSE_RF, RMSE_RF,cvRMSE_RF,NMBE_RF= evaluating_models(Y_test,y_pred_RF)

        fig.text(0.5, 0.02, f'Mean Absolute Error: {round(MAE_RF,4)}   Mean Bias Error: {round(MBE_RF,4)}   Mean Squared Error: {round(MSE_RF,4)}    Root Mean Squared Error: {round(RMSE_RF,4)}   cvRMSE: {round(cvRMSE_RF,4)}  Normalized Mean Bias Error: {round(NMBE_RF,4)}', ha='center', va='center', fontsize=12, color='black')

        plt.show()
        if option==high_feat_temp_co[f"high_feat_temp_{country}"]: 
            plt.title(f'CO2 emission training for {country} with RF')
            plt.savefig(f"regressions/co2/rf_co2_{country}.png")   
            
        elif option==high_feat_temp_en[f"high_feat_temp_{country}"]:
            plt.title(f'Renewable energy training for {country} with RF')
            plt.savefig(f"regressions/energy/rf_en_{country}.png") 
        plt.clf()

        ######################################################## RF With Uniformized data ##################################################################################################################


        scaler = StandardScaler()
        # Fit only to the training data
        scaler.fit(X_train)

        # Now apply the transformations to the data:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        parameters = {'bootstrap': True,
                    'min_samples_leaf': 4,
                    'n_estimators': 1030, 
                    'min_samples_split': 2,
                    'max_features': "sqrt",
                    'max_depth': 70,
                    'max_leaf_nodes': None}

        RF_model_uni= RandomForestRegressor(**parameters)


        RF_model_uni.fit(X_train_scaled, Y_train.reshape(-1,1))
        y_pred_RF_uni = RF_model_uni.predict(X_test_scaled)

        plt.subplot(1, 2, 1)

        plt.plot(Y_test, label="Test Data")
        plt.plot(y_pred_RF_uni, label="Predicted Data")
        plt.show()
        plt.subplot(1, 2, 2)
        plt.scatter(Y_test,y_pred_RF_uni)
        plt.show()

        MAE_RF, MBE_RF,MSE_RF, RMSE_RF,cvRMSE_RF,NMBE_RF= evaluating_models(Y_test,y_pred_RF_uni)

        fig.text(0.5, 0.02, f'Mean Absolute Error: {round(MAE_RF,4)}   Mean Bias Error: {round(MBE_RF,4)}   Mean Squared Error: {round(MSE_RF,4)}    Root Mean Squared Error: {round(RMSE_RF,4)}   cvRMSE: {round(cvRMSE_RF,4)}  Normalized Mean Bias Error: {round(NMBE_RF,4)}', ha='center', va='center', fontsize=12, color='black')

        plt.show()
        plt.title(f'{country}')
        
        if option==high_feat_temp_co[f"high_feat_temp_{country}"]: 
            plt.title(f'CO2 emission training for {country} with RF Uni')
            plt.savefig(f"regressions/co2/rf_uniform_co2_{country}.png")   
            
        elif option==high_feat_temp_en[f"high_feat_temp_{country}"]:
            plt.title(f'Renewable energy training for {country} with RF Uni')
            plt.savefig(f"regressions/energy/rf_uniform_en_{country}.png") 
            
        plt.clf()
        
        
    ################################################### DATA TESTE ####################################################################################################
        
        print(country)
        df_test= pd.read_csv(f'test_data/{country}_Test.csv', index_col = False)

        date_range= pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        print(len(date_range))  # Check the length of date_range
        print(len(df_test)) 
        
        df_test.index = date_range

        df_test.index.name="Date"
        
        df_test.rename(columns={'temp': 'Temperature [ºC]', 'windspeed': 'Wind Speed [km/h]',"humidity": 'Relative Humidity (%)',
                                    'sealevelpressure': 'Pressure [mbar]','solarradiation': 'Solar Radiation [W/m^2]', 'precip': 'Rain [mm/h]'}, inplace=True)

        
        df_carbon = carbon_eu[(carbon_eu['country'] == country)] 
        
        for sector in sectors_co2:
            sector_list = df_carbon[(df_carbon['sector'] == sector)]
            sector_list = sector_list.drop(columns=['timestamp','sector'])
            sector_list.rename(columns={'value': f'{sector}', 'date': 'Date'}, inplace=True)
            globals()[f"{sector.replace(' ', '_')}"]= sector_list 

        df_carbon = pd.merge(Power, Ground_Transport, on=["Date","country"], how='inner')

        # Merge all types of energy 

        df_carbon  = pd.merge(df_carbon , International_Aviation, on=["Date","country"], how='inner')

        df_carbon  = pd.merge(df_carbon , Residential, on=["Date","country"], how='inner')

        df_carbon = pd.merge(df_carbon , Industry, on=["Date","country"], how='inner')

        df_carbon  = pd.merge(df_carbon , Domestic_Aviation, on=["Date","country"], how='inner')

        
        df_carbon["Total CO2 [Tonne]"]= df_carbon["Power"]+df_carbon["International Aviation"]+df_carbon["Residential"]+df_carbon["Industry"]+df_carbon["Domestic Aviation"]

        df_carbon[['Day','Month','Year']] = df_carbon.Date.str.split("/",expand=True) 

        df_carbon = df_carbon.drop(columns=['Date'])

        df_carbon['Date'] = pd.to_datetime(df_carbon[['Day', 'Month', 'Year']])

        df_carbon = df_carbon.drop(columns=['Year', 'Month', 'Day'])

        df_carbon.set_index('Date', inplace=True)
        
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
        
        df_test  = pd.merge(df_test , df_carbon, on=["Date"], how='inner')
        
        df_test  = pd.merge(df_test, df_en, on=["Date","country"], how='inner')
        
        df_test=df_test.dropna() 
        
        df_list=[]
        
        flights = pd.read_excel(f"flights_data/2023flight.xlsx", sheet_name="Data")
        
        flights['Year'] = flights['Day'].dt.year
        flights['Month'] = flights['Day'].dt.month
        flights['Day'] = flights['Day'].dt.day

        flights['Date'] = pd.to_datetime(flights[['Year', 'Month', 'Day']])

        flights = flights.drop(columns=['Year','Week', 'Month', 'Day','Flights (7-day moving average)','Day 2019', 'Flights 2019 (Reference)',	'% vs 2019 (Daily)','% vs 2019 (7-day Moving Average)',	'Day Previous Year',	'Flights Previous Year'])

        flights.set_index('Date', inplace=True)
        
        flights.rename(columns={'Entity':'country'}, inplace=True)

        
        df_test  = pd.merge(df_test , flights, on=["Date","country"], how='inner')
        
        if "datetime" in df_test:
            df_test = df_test.drop(columns=['datetime'])
            
        print(df_test)
        
        df_test['CO2 Emission-1 [kW]']=df_test['Total CO2 [Tonne]'].shift(1) # Previous hour consumption

        df_test["3 Last CO2 Mean"] = df_test['Total CO2 [Tonne]'].rolling(window=4).apply(lambda x: x[-4:-1].mean(), raw=True) #mean of the last three consumption values for each index

        df_test['Ren Energy-1 [kW]']=df_test['Total Renewable [GWh]'].shift(1) # Previous hour consumption

        df_test["3 Last Ren Energy Mean"] = df_test['Total Renewable [GWh]'].rolling(window=4).apply(lambda x: x[-4:-1].mean(), raw=True) #mean of the last three consumption values for each index
        
        df_test=df_test.dropna() 
        
        ########################################################### FORECASTING PARA 2023 #######################################################################################################################
        
        forecast_test=pd.DataFrame()

        if option==high_feat_temp_co[f"high_feat_temp_{country}"]:   
            forecast_test["Total CO2 [Tonne]"]=df_test["Total CO2 [Tonne]"]
        elif option==high_feat_temp_en[f"high_feat_temp_{country}"]:
            forecast_test["Total Renewable [GWh]"]=df_test["Total Renewable [GWh]"]
        
        for feat in option:
            forecast_test[feat[0]]=df_test[f"{feat[0]}"]
            
        Z_test=forecast_test.values
        
        Y_test=Z_test[:, 0]
        #Identify input Y
        X_test=Z_test[:,[1,2,3,4,5,6]]  # Temperature,Solar Radiation, Power-1, average value of last three data values

        y_pred_RF_uni = RF_model_uni.predict(scaler.transform(X_test))
        
        plt.subplot(1, 2, 1)

        plt.plot(Y_test)
        plt.plot(y_pred_RF_uni)
        plt.show()

        plt.subplot(1, 2, 2)

        plt.scatter(Y_test,y_pred_RF_uni)
        plt.show()

        MAE_RF, MBE_RF,MSE_RF, RMSE_RF,cvRMSE_RF,NMBE_RF= evaluating_models(Y_test,y_pred_RF_uni)

        fig.text(0.5, 0.02, f'Mean Absolute Error: {round(MAE_RF,4)}   Mean Bias Error: {round(MBE_RF,4)}   Mean Squared Error: {round(MSE_RF,4)}    Root Mean Squared Error: {round(RMSE_RF,4)}   cvRMSE: {round(cvRMSE_RF,5)}  Normalized Mean Bias Error: {round(NMBE_RF,5)}', ha='center', va='center', fontsize=12, color='black')

        plt.show()

        if option==high_feat_temp_co[f"high_feat_temp_{country}"]:  
            plt.title(f'CO2 Forecast for {country} with RF Uni') 
            plt.savefig(f"forecast/co2/forecast_co2_{country}.png") 
        elif option==high_feat_temp_en[f"high_feat_temp_{country}"]:
            plt.title(f'Renewable energy Forecast for {country} with RF Uni')
            plt.savefig(f"forecast/energy/forecast_en_{country}.png") 
        
        plt.clf()
        

    
            
            
            
                
                    

