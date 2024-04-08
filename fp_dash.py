import dash
import traceback
from dash import html, dcc
from dash.dependencies import Input, Output, State
import base64
import smtplib
from email.mime.text import MIMEText
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import pickle
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
import numpy as np
import plotly.graph_objects as go
import os
import urllib.request
import re

external_stylesheets = ["https://www.w3schools.com/w3css/4/w3.css"]  # CSS Style imported

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df_total_country = pd.read_csv('csv_files/Allcountries.csv')

print(df_total_country.columns)
app.layout = html.Div([
        dcc.Graph(id="map1"),
                    html.Label("CO2 emissions setors:"),
                    dcc.Dropdown(id="mydropdown1",
                                 options= [
                                        'Total CO2 [Tonne]',
                                        'Ground Transport',
                                        'International Aviation',
                                        'Residential',
                                        'Industry',
                                        'Domestic Aviation'
                                    ],
                                 value="Total CO2 [Tonne]"),
        dcc.Graph(id="map2"),
                    html.Label("Energy setors:"),
                    dcc.Dropdown(id="mydropdown2",
                                 options= [
                                        'Other sources',
                                        'Gas',
                                        'Oil',
                                        'Coal',
                                        'Wind',
                                        'Nuclear',
                                        'Solar',
                                        'Hydroelectricity',
                                        'Total Renewable [GWh]',
                                        'Total Non-Renewable [GWh]',
                                        'Total Electricity [GWh]'
                                    ],
                                 value="Total Electricity [GWh]"),
                    
        dcc.Graph(id="map3"),
                    html.Label("Climate options"),
                    dcc.Dropdown(id="mydropdown3",
                                 options= [
                                        'Temperature [ºC]',
                                        'Relative Humidity (%)',
                                        'Rain [mm/h]',
                                        'Wind Speed [km/h]',
                                        'Pressure [mbar]',
                                        'Solar Radiation [W/m^2]'
                                    ],
                                 value="Temperature [ºC]"),

])


@app.callback(
    Output('map1', 'figure'),
    Input('mydropdown1', 'value')
)

def display_choropleth1(variable1):
    fig1 = px.choropleth(df_total_country,
                        geojson="https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson",
                        featureidkey='properties.NAME',
                        locations="country",
                        animation_frame='Date',  # column in dataframe
                        color=variable1,  # dataframe
                        color_continuous_scale='OrRd',
                        title=f"{variable1} in Europe",
                        height=700
                        )
    fig1.update_geos(fitbounds="locations", visible=False)
    return fig1

@app.callback(
    Output('map2', 'figure'),
    Input('mydropdown2', 'value')
)

def display_choropleth2(variable2):
    fig2 = px.choropleth(df_total_country,
                        geojson="https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson",
                        featureidkey='properties.NAME',
                        locations="country",
                        animation_frame='Date',  # column in dataframe
                        color=variable2,  # dataframe
                        color_continuous_scale='Mint',
                        title=f"{variable2} in Europe",
                        height=700
                        )
    fig2.update_geos(fitbounds="locations", visible=False)
    return fig2

@app.callback(
    Output('map3', 'figure'),
    Input('mydropdown3', 'value')
)


def display_choropleth3(variable3):
    fig3 = px.choropleth(df_total_country,
                        geojson="https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson",
                        featureidkey='properties.NAME',
                        locations="country",
                        animation_frame='Date',  # column in dataframe
                        color=variable3,  # dataframe
                        color_continuous_scale='Magenta',
                        title=f"{variable3} in Europe",
                        height=700
                        )
    fig3.update_geos(fitbounds="locations", visible=False)
    return fig3


if __name__ == '__main__':
    app.run_server(debug=True)