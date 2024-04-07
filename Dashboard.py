import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
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

#Define CSS style
external_stylesheets = ["https://www.w3schools.com/w3css/4/w3.css"]

#########################################################################################















##########################################################################################

forecast_dropdown = [
    "CO2 Emissions", 
    "Linear Regression", 
    "Gradient Boosting"
    ]

countries=["Portugal","Spain","France","Germany","Belgium","Italy"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Define the layout of the app
app.layout = html.Div(
    style={'backgroundColor': '#fbfbfb', "margin": "0", "padding": "0"},
    children=[
    html.H1("Project 3: Dashboard", style={"margin-left": "10px", 'textAlign': 'center'}),
    
    # Power Forecast
    html.Div([
        html.H2("Forecasting", style={"margin-left": "10px"}),
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id="forecast-dropdown",
                    options=[{"label": metric, "value": metric} for metric in forecast_dropdown],
                    value=forecast_dropdown[0],
                    clearable=False
                ),
            ], style={"width": "200px", "display": "inline-block", "margin-right": "10px"}),
            html.Div([
                dcc.Dropdown(
                    id="country-dropdown",
                    options=[{"label": graph_type, "value": graph_type} for country in countries],
                    value=countries[0],
                    clearable=False
                ),
            ], style={"width": "200px", "display": "inline-block"}),
            dcc.Graph(id="forecast-graph")
        ]),

    ]),
    
])

def generate_table(dataframe, max_rows=20):
    table_style = {
        'borderCollapse': 'collapse',
        'width': '100%',
        'border': '1px solid black',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '14px',
        'maxWidth': '500px'
    }

    header_style = {
        'backgroundColor': '#efefef',
        'border': '1px solid black',
        'padding': '8px'
    }

    cell_style = {
        'border': '1px solid black',
        'padding': '8px'
    }
    
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col, style=header_style) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(
                    dataframe.iloc[i][col], style=cell_style) for col in dataframe.columns], 
                    style={'backgroundColor': 'white' if i % 2 == 0 else '#f8f8f8'}
                ) for i in range(min(len(dataframe), max_rows))]
            )
        ],
        style=table_style
    )

@app.callback(Output("forecast-graph", "figure"),
             [Input("forecast-dropdown", "value"), Input("country-dropdown", "value")])

def update_forecast_graph(selected_metric, selected_country):
    if selected_tab == "Bagging Regression":
        fig = fig_forecast_BR
    elif selected_tab == "Linear Regression":
        fig = fig_forecast_LR
    elif selected_tab == "Gradient Boosting":
        fig = fig_forecast_GB
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
