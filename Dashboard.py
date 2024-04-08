import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import pandas as pd # data science library o manipulate data
import numpy as np # mathematical library to manipulate arrays and matrices
import matplotlib.pyplot as plt # visualization ~
#from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn import  metrics
import statsmodels.api as sm
import plotly.express as px

import MetricForecast

#Define CSS style
external_stylesheets = ["https://www.w3schools.com/w3css/4/w3.css"]

#########################################################################################

df_total_country = pd.read_csv('csv_files/Allcountries.csv')













##########################################################################################

forecast_dropdown = [
    "CO2 Emissions", 
    "Renewable Energy", 
    ]

metrics = ["MAE", "MBE", "MSE", "RMSE", "NMBE", "cvRMSE"]

countries = MetricForecast.countries

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Define the layout of the app

app.layout = html.Div(
    children=[
        html.Title("W3.CSS Template"),
        html.Meta(charSet="UTF-8"),
        html.Meta(name="viewport", content="width=device-width, initial-scale=1"),
        html.Link(rel="stylesheet", href="https://www.w3schools.com/w3css/4/w3.css"),
        html.Link(
            rel="stylesheet", href="https://fonts.googleapis.com/css?family=Poppins"
        ),
        
        # Sidebar Menu
        html.Nav(
            className="w3-sidebar w3-blue w3-collapse w3-top w3-large w3-padding",
            style={"z-index": "3", "width": "300px", "font-weight": "bold"},
            id="mySidebar",
            children=[
                html.Br(),
                html.A(
                    href="#",
                    className="w3-button w3-hide-large w3-display-topleft",
                    style={"width": "100%", "font-size": "22px"},
                    children=["Close Menu"],
                ),
                html.Div(
                    className="w3-container",  # Creating several buttons to construct the side menu
                    children=[
                        html.H3(
                            className="w3-padding-64",
                            children=[
                                html.B(
                                    children=[
                                        "Energy",
                                        html.Br(),
                                        html.Span(children=["Services"]),
                                    ]
                                )
                            ],
                        )
                    ],
                ),
                html.Div(
                    className="w3-bar-block",
                    children=[
                        html.A(
                            href="#showcase",
                            className="w3-bar-item w3-button w3-hover-white",
                            children=["Introduction"],
                        ),
                        html.A(
                            href="#maps",
                            className="w3-bar-item w3-button w3-hover-white",
                            children=["European Maps"],
                        ),
                        html.A(
                            href="#data_analysis",
                            className="w3-bar-item w3-button w3-hover-white",
                            children=["Data Analysis"],
                        ),
                        html.A(
                            href="#features",
                            className="w3-bar-item w3-button w3-hover-white",
                            children=["Features"],
                        ),
                        html.A(
                            href="#forecast",
                            className="w3-bar-item w3-button w3-hover-white",
                            children=["Forecast and Metrics"],
                        ),
                        html.A(
                            href="#contacts",
                            className="w3-bar-item w3-button w3-hover-white",
                            children=["Contacts"],
                        ),
                    ],
                ),
            ],
        ),
        
        html.Header(
            className="w3-container w3-top w3-hide-large w3-blue w3-xlarge w3-padding",
            children=[
                html.A(
                    href="#",
                    className="w3-button w3-blue w3-margin-right",
                    children=["☰"],
                ),
                html.Span(children=["Company Name"]),
            ],
        ),
        html.Div(
            className="w3-overlay w3-hide-large",
            style={"cursor": "pointer"},
            title="close side menu",
            id="myOverlay",
        ),
        html.Div(
            className="w3-main",
            style={"margin-left": "340px", "margin-right": "40px"},
            children=[
                html.Div(
                    className="w3-container",
                    style={"margin-top": "80px"},
                    id="showcase",
                    children=[
                        html.H1(
                            className="w3-xxxlarge",
                            children=[
                                html.B(["European CO2 Emission and Energy Consumption"])
                            ],
                        ),
                        html.H1(
                            className="w3-xlarge w3-text-blue",
                            children=[html.B(["Introduction."])],
                        ),
                        html.Hr(
                            style={
                                "width": "50px",
                                "border": "5px solid rgba(0, 128, 255, 0.904)",
                            },
                            className="w3-round",
                        ),
                        html.P(
                    children=[
                        """
                        Welcome to the European CO2 Emission and Energy Consumption Analyzer. 
                        This dashboard provides the user with information on the CO2 emissions and energy consumption of all European Union countries, including data for each sector and meteorological information from 01-01-2021 to 01-01-2023. 
                        Users can also perform data analysis on a global level or for a specific country, and forecast CO2 emissions and energy consumption for individual countries between 01-01-2023 and 31-12-2023. To select a country for forecasting, simply click on the country of interest on the first map. Portugal is the default country.
                        """
                        ]
                        ),
                    ],
                ),
                
        html.Div( #starting the part of data analysis 
                    className="w3-container",
                    id="maps",
                    style={"margin-top": "75px"},
                    children=[
                        html.H1(
                            className="w3-xlarge w3-text-blue",
                            children=[html.B(["European Maps."])],
                        ),
                        html.Hr(
                            style={
                                "width": "50px",
                                "border": "5px solid rgba(0, 128, 255, 0.904)",
                            },
                            className="w3-round",
                        ),
                        ],
                    ),
        html.Div([
            dcc.Graph(id="map1",clickData={'points': [{'location': 'Portugal'}]}),
            html.Label("CO2 emissions sectors:"),
            dcc.Dropdown(
                id="mydropdown1",
                options= [
                        'Total CO2 [Tonne]',
                        'Ground Transport',
                        'International Aviation',
                        'Residential',
                        'Industry',
                        'Domestic Aviation'
                    ],
                value="Total CO2 [Tonne]"
            ),

            dcc.Graph(id="map2"),
            html.Label("Energy sectors:"),
            dcc.Dropdown(
                id="mydropdown2",
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
                value="Total Electricity [GWh]"
            ),

            dcc.Graph(id="map3"),
            html.Label("Climate options"),
            dcc.Dropdown(
                id="mydropdown3",
                options= [
                            'Temperature [ºC]',
                            'Relative Humidity (%)',
                            'Rain [mm/h]',
                            'Wind Speed [km/h]',
                            'Pressure [mbar]',
                            'Solar Radiation [W/m^2]'
                        ],
                value="Temperature [ºC]"
            ),
        ]),
        
        
        
        html.Div( #starting the part of data analysis 
                    className="w3-container",
                    id="forecast",
                    style={"margin-top": "75px"},
                    children=[
                        html.H1(
                            className="w3-xlarge w3-text-blue",
                            children=[html.B(["Forecast and Metrics."])],
                        ),
                        html.Hr(
                            style={
                                "width": "50px",
                                "border": "5px solid rgba(0, 128, 255, 0.904)",
                            },
                            className="w3-round",
                        ),
                        ],
                    ),
        html.Div([
                html.Div([
                    dcc.Dropdown(
                        id="forecast-dropdown",
                        options=[{"label": observable, "value": observable} for observable in forecast_dropdown],
                        value=forecast_dropdown[0],
                        clearable=False
                    ),
                ], style={"width": "200px", "display": "inline-block", "margin-right": "10px"}),
                html.Div([html.Div([html.P(id="clicked-location")]),
                dcc.Graph(id="forecast-graph"),
                html.H4("Metrics", style={"margin-left": "10px"}),
                dcc.Checklist(
                    id="metrics-checklist",
                    options=[{"label": metric, "value": metric} for metric in metrics],
                    value=metrics,
                    inline=True, 
                    labelStyle={"display": "inline-block", "margin-right": "50px"}  
                ),
                html.Div(id="metrics-table", style={"margin-top": "20px"}),
            ]),
        ]),
    ]
)
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

    # Modify the function to format numeric values to 5 decimal places
    formatted_data = dataframe.applymap(lambda x: "{:.5f}".format(x) if isinstance(x, (int, float)) else x)

    return html.Table([
        html.Thead(
            html.Tr([html.Th(col, style=header_style) for col in formatted_data.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(
                    formatted_data.iloc[i][col], style=cell_style) for col in formatted_data.columns],
                    style={'backgroundColor': 'white' if i % 2 == 0 else '#f8f8f8'}
                ) for i in range(min(len(formatted_data), max_rows))]
            )
        ],
        style=table_style
    )

    
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

@app.callback(
    Output('clicked-location', 'children'),
    [Input('map1', 'clickData')]
)

def update_clicked_location(click_data1):
    print("Map 1:", click_data1)
    return f"You have selected {click_data1['points'][0]['location']} to perform the forecast."



@app.callback([Output("forecast-graph", "figure"), Output("metrics-table", "children")],
             [Input("forecast-dropdown", "value"), Input('map1', 'clickData'), Input("metrics-checklist", "value")])

def update_forecast(selected_observable, selected_country, selected_metrics):
    df_plot, df_metrics = MetricForecast.Forecast(selected_country['points'][0]['location'], selected_observable)
    df_plot.rename(columns={"y_pred": "Forecast"}, inplace=True)
    df_test = MetricForecast.df_test_dict[selected_country['points'][0]['location']]
    if selected_observable == "CO2 Emissions":
        df_plot["Real Data"] = df_test["Total CO2 [Tonne]"]
    elif selected_observable == "Renewable Energy":
        df_plot["Real Data"] = df_test["Total Renewable [GWh]"]
        
    fig = px.line(df_plot, y=df_plot.columns[[0,1]])
    fig.update_yaxes(title_text=selected_observable)
    table = generate_table(df_metrics.loc[:,selected_metrics])
    
    return fig, table


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
