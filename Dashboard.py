import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import pandas as pd # data science library o manipulate data
import plotly.express as px
from datetime import date
import traceback
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import plotly.graph_objects as go

import MetricForecast

#Define CSS style
external_stylesheets = ["https://www.w3schools.com/w3css/4/w3.css"]

df_total_country = pd.read_csv('csv_files/Allcountries.csv')

map_tabs = [
    "CO2 Emissions",
    "Energy",
    "Meteorology"
    ]

CO2_map_dropdown = [
    "Total CO2 [Tonne]",
    "Power",
    "Ground Transport",
    "International Aviation",
    "Residential",
    "Industry",
    "Domestic Aviation"
    ]

Energy_map_dropdown = [
    "Nuclear",
    "Gas",
    "Oil",
    "Coal",
    "Wind",
    "Solar",
    "Hydroelectricity",
    "Other sources",
    "Total Renewable [GWh]",
    "Total Non-Renewable [GWh]",
    "Total Electricity [GWh]"
    ]

Meteo_map_dropdown= [
    "Temperature [ºC]",
    "Relative Humidity (%)",
    "Rain [mm/h]",
    "Wind Speed [km/h]",
    "Pressure [mbar]",
    "Solar Radiation [W/m^2]"
    ]

forecast_dropdown = [
    "CO2 Emissions", 
    "Renewable Energy", 
    ]

metrics = ["MAE", "MBE", "MSE", "RMSE", "NMBE", "cvRMSE"]

countries = MetricForecast.countries

df_population = pd.read_csv("population_data.csv")
df_population = df_population[["Country Name","2021","2022"]]
df_population.set_index("Country Name", inplace=True)

click_data = {'points': [{'location': 'Portugal'}]}


def period_anal(start_date, end_date, variable):
    period_sum = []
    for country in countries:
        # Filter the DataFrame based on the conditions: date range and country
        df_filter = df_total_country[(df_total_country['Date'] >= start_date) & 
                                     (df_total_country['Date'] <= end_date) & 
                                     (df_total_country['country'] == country)]
        # Sum the values for the selected variable in the filtered DataFrame
        period_sum.append(df_filter[variable].sum())
    return period_sum

def period_anal_per_capita(start_date, end_date, variable):
    period_sum = []
    for country in countries:
        # Filter the DataFrame based on the conditions: date range and country
        df_filter = df_total_country[(df_total_country['Date'] >= start_date) & 
                                     (df_total_country['Date'] <= end_date) & 
                                     (df_total_country['country'] == country)]
        #Assuming the population hasn't changed much in 2 years
        population = (df_population.loc[country, "2021"] + df_population.loc[country, "2022"])/2
        period_sum.append(df_filter[variable].sum()/population)
    return period_sum
        
def data_analysis_sing(country):
    df = MetricForecast.get_country_variable(country)
    return df

def get_features(country):
    kbest_co, kbest_en, rfor_co, rfor_en, points_co_dic, points_en_dic= MetricForecast.get_features(country)
    
    return kbest_co, kbest_en, rfor_co, rfor_en, points_co_dic, points_en_dic

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
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
                            children=["Interactive Map"],
                        ),
                        html.A(
                            href="#data_analysis",
                            className="w3-bar-item w3-button w3-hover-white",
                            children=["Global Data Analysis"],
                        ),
                        html.A(
                            href="#data_analysis_country",
                            className="w3-bar-item w3-button w3-hover-white",
                            children=["Country Data Analysis"],
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
                            children=[html.B(["Introduction"])],
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
                            children=[html.B(["Interactive Map"])],
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
            dcc.Tabs(id="map-tabs",
                 children=[dcc.Tab(label=tab, value=tab) for tab in map_tabs],
                 value=map_tabs[0], 
            ),
            #Dummy Graphs to avoid callback errors when the dashboard is launched
            dcc.Graph(id="co2-map", style={"display": "none"}),
            dcc.Graph(id="energy-map", style={"display": "none"}),
            dcc.Graph(id="meteo-map", style={"display": "none"}),
            html.Div(id="map")
        ]),
        
        html.Div( #starting the part of data analysis 
                    className="w3-container",
                    id="data_analysis",
                    style={"margin-top": "75px"},
                    children=[
                        html.H1(
                            className="w3-xlarge w3-text-blue",
                            children=[html.B(["Global Data Analysis"])],
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
        html.Div([
            dcc.DatePickerRange(
                id='my-date-picker-range',
                min_date_allowed=date(2021, 1, 1),
                max_date_allowed=date(2023, 1, 1),
                initial_visible_month=date(2021, 1, 1),
                start_date=date(2021, 1, 1),
                end_date=date(2023, 1, 1)
            ),
        ], style = {"width": "600px"}),
        html.Div(id='output-container-date-picker-range'),
        dcc.Dropdown(
            id="option-dropdown",
            options=[
                {'label': 'Total CO2 [Tonne]', 'value': 'Total CO2 [Tonne]'},
                {'label': 'Total CO2 per Capita [g]', 'value': 'Total CO2 per Capita [g]'},
                {'label': 'Total Renewable [GWh]', 'value': 'Total Renewable [GWh]'},
                {'label': 'Total Non-Renewable [GWh]', 'value': 'Total Non-Renewable [GWh]'},
                {'label': 'Total Electricity [GWh]', 'value': 'Total Electricity [GWh]'},
                {'label': 'Carbon Intensity of Energy Production [Tonne/GWh]', 'value': 'Carbon Intensity of Energy Production [Tonne/GWh]'},
                {'label': 'Renewable Energy Ratio [%]', 'value': 'Renewable Energy Ratio [%]'},
            ],
            value='Total CO2 [Tonne]',
            clearable=False
           
        ),
    ],),
    html.Br(),
    html.Br(),
    dcc.Graph(id="period-global-bar")
]),
        
        
        html.Div( #starting the part of data analysis 
                    className="w3-container",
                    id="data_analysis_country",
                    style={"margin-top": "75px"},
                    children=[
                        html.H1(
                            className="w3-xlarge w3-text-blue",
                            children=[html.B(["Country Data Analysis"])],
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
                        id="sector_dropdown",
                        options=["CO2 Emissions", "Energy Consumption", "Climate data"],
                        value="CO2 Emissions",
                        clearable=False
                    ),
                ], style={"width": "200px", "display": "inline-block", "margin-right": "10px"}),
                html.Div([html.Div([html.P(id="clicked-location2")]),
                dcc.Graph(id="sector_graph"),
                dcc.Graph(id="pie-graph"),
            ]),
        ]),
        
        html.Div([
                html.Div([
                    dcc.Dropdown(
                        id="correlation_dropdown",
                        options=["CO2 Emission and Temperature", 
                                 "Energy Production and Temperature", 
                                 "CO2 Emission and Total Energy Produced",
                                 "Renewable Energy and Total Energy Produced"
                                 ],
                        value="CO2 Emission and Temperature",
                        clearable=False,
                        style={"width": "400px"}
                    ),                   
                ], style={"width": "200px", "display": "inline-block", "margin-right": "10px"}),
                html.Div([html.Div([html.P(id="clicked-location3")]),
                dcc.Graph(id="correlation-graph"),
            ]),
        ]),
    
        html.Div( #starting the part of data analysis 
                    className="w3-container",
                    id="features",
                    style={"margin-top": "75px"},
                    children=[
                        html.H1(
                            className="w3-xlarge w3-text-blue",
                            children=[html.B(["Features"])],
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
        html.P(
                    children=[
                        """
                        The variables for each country are evaluated using kBest and Random Forest Regressor tests, and scored according to their importance in a global view of both tests to perform the forecast.
                        """
                        ]
                        ),
                html.Div([
                html.Div([
                    dcc.Dropdown(
                        id="fore_dropdown",
                        options=["CO2 Emission", "Renewable Energy"],
                        value="CO2 Emission",
                        clearable=False,
                    ),], style={"width": "200px", "display": "inline-block", "margin-right": "10px"}),
                
                html.Div([
                    dcc.Dropdown(
                        id="feature_dropdown",
                        options=["kBest", "Random Forest Regressor"],
                        value="kBest",
                        clearable=False
                    ),
                 ], style={"width": "200px", "display": "inline-block"}),
                ]),
               
                    html.Div([html.P(id="clicked-location4")]),
                          
            html.Div([
                dcc.Graph(id="features_bar"),
                dcc.Graph(id="score_bar"),
                html.Label("Note that this 'Final Score' is given based on each feature's performance on both tests. Those with higher scores on this graph are the chosen features to use in the forecast."),
            ]),
        
        
        
        html.Div( #starting the part of data analysis 
                    className="w3-container",
                    id="forecast",
                    style={"margin-top": "75px"},
                    children=[
                        html.H1(
                            className="w3-xlarge w3-text-blue",
                            children=[html.B(["Forecast and Metrics"])],
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
        html.Div( #Starting a little pessoal contacts page with the option of sending feedbacks (it is going to an email that will be given in the classroom private comentaries)
                    className="w3-container",
                    id="contacts",
                    style={"margin-top": "75px"},
                    children=[
                        html.H1(
                            className="w3-xlarge w3-text-blue",
                            children=[html.B(["Contacts"])],
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
                                "If you need some help or have some suggestions for new features feel free to contact us."
                            ]
                        ),
                        html.Form(
                            action="/action_page.php",
                            target="_blank",
                            children=[
                                html.Div(
                                    className="w3-section",
                                    children=[html.Label("Name"),": Gonçalo Martins, Rafael Parente, João Santos"],
                                                                 
                                ),

                                html.Div(
                                    className="w3-section",
                                    children=[html.Label("Let us know your feedback in the box below!")],
                            
                                ),
                                html.Br(),
                                html.Div([
                                dcc.Textarea(
                                    id='email-body',
                                    placeholder='Compose your email here...',
                                    style={'width': '100%', 'height': '200px'}
                                ),
                                html.Button(
                                    type="button",
                                     id='send-button',
                                     n_clicks=0,
                                    className="w3-button w3-block w3-padding-large w3-blue w3-margin-bottom",
                                    children=["Send Message"],
                                ),
                                html.Div(id='email-status')
                            ]),
                                
                                
                            ],
                        ),
                    ],
                ),
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
        'maxWidth': '600px'
    }

    header_style = {
        'backgroundColor': '#efefef',
        'border': '1px solid black',
        'padding': '8px'
    }

    cell_style = {
        'border': '1px solid black',
        'padding': '8px',
        'textAlign': 'center'
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

@app.callback(Output("map", "children"),
              Input("map-tabs", "value"))

def render_content(selected_tab):
    if selected_tab == "CO2 Emissions":
        return html.Div([
            dcc.Graph(id="co2-map", clickData=click_data),
            html.Label("CO2 emissions sectors:"),
            dcc.Dropdown(
                id="co2-map-dropdown",
                options= [{"label": observable, "value": observable} for observable in CO2_map_dropdown],
                value=CO2_map_dropdown[0],
                clearable=False
                ),
            dcc.Graph(id="energy-map", style={"display": "none"}),
            dcc.Graph(id="meteo-map", style={"display": "none"})
        ])
    elif selected_tab == "Energy":
        return html.Div([
            dcc.Graph(id="energy-map", clickData=click_data),
            html.Label("Energy sectors:"),
            dcc.Dropdown(
                id="energy-map-dropdown",
                options= [{"label": observable, "value": observable} for observable in Energy_map_dropdown],
                value=Energy_map_dropdown[0],
                clearable=False
            ),
            dcc.Graph(id="co2-map", style={"display": "none"}),
            dcc.Graph(id="meteo-map", style={"display": "none"})
        ])
    elif selected_tab == "Meteorology":
        return html.Div([
            dcc.Graph(id="meteo-map", clickData=click_data),
            html.Label("Climate options:"),
            dcc.Dropdown(
                id="meteo-map-dropdown",
                options= [{"label": observable, "value": observable} for observable in Meteo_map_dropdown],
                value=Meteo_map_dropdown[0],
                clearable=False
            ),
            dcc.Graph(id="co2-map", style={"display": "none"}),
            dcc.Graph(id="energy-map", style={"display": "none"})
        ])

    
@app.callback(
    Output('co2-map', 'figure'),
    Input('co2-map-dropdown', 'value')
)

def display_choropleth_co2(variable):
    fig = px.choropleth(df_total_country,
                        geojson="https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson",
                        featureidkey='properties.NAME',
                        locations="country",
                        animation_frame='Date',  # column in dataframe
                        color=variable,  # dataframe
                        color_continuous_scale='OrRd',
                        title=f"{variable} in Europe",
                        height=700
                        )
    fig.update_geos(fitbounds="locations", visible=False)
    return fig

@app.callback(
    Output('energy-map', 'figure'),
    Input('energy-map-dropdown', 'value')
)

def display_choropleth_energy(variable):
    fig = px.choropleth(df_total_country,
                        geojson="https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson",
                        featureidkey='properties.NAME',
                        locations="country",
                        animation_frame='Date',  # column in dataframe
                        color=variable,  # dataframe
                        color_continuous_scale='Mint',
                        title=f"{variable} in Europe",
                        height=700
                        )
    fig.update_geos(fitbounds="locations", visible=False)
    return fig

@app.callback(
    Output('meteo-map', 'figure'),
    Input('meteo-map-dropdown', 'value')
)


def display_choropleth_meteo(variable):
    fig = px.choropleth(df_total_country,
                        geojson="https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson",
                        featureidkey='properties.NAME',
                        locations="country",
                        animation_frame='Date',  # column in dataframe
                        color=variable,  # dataframe
                        color_continuous_scale='Magenta',
                        title=f"{variable} in Europe",
                        height=700
                        )
    fig.update_geos(fitbounds="locations", visible=False)
    return fig

@app.callback(Output('clicked-location', 'children'),
              [Input("co2-map", "clickData"), 
               Input("energy-map", "clickData"), 
               Input("meteo-map", "clickData")])

def update_clicked_location(click_data_co2, click_data_energy, click_data_meteo):
    global click_data
    if click_data_co2:
        click_data = click_data_co2
        return f"You have selected {click_data_co2['points'][0]['location']} to perform the forecast."
    elif click_data_energy:
        click_data = click_data_energy
        return f"You have selected {click_data_energy['points'][0]['location']} to perform the forecast."
    elif click_data_meteo:
        click_data = click_data_meteo
        return f"You have selected {click_data_meteo['points'][0]['location']} to perform the forecast."
    elif click_data:
        return f"You have selected {click_data['points'][0]['location']} to perform the forecast."
    else:
        return "No country selected for forecast."


@app.callback(Output("period-global-bar", "figure"),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    Input("option-dropdown", "value"))

def graph_bar_global(start_date,end_date, variable):
    if variable == "Total CO2 per Capita [g]":
        sum_period1 = period_anal_per_capita(start_date,end_date,"Total CO2 [Tonne]")
        sum_period = [a*1000000 for a in sum_period1]
    elif variable == "Carbon Intensity of Energy Production [Tonne/GWh]":
        sum_period1 = period_anal(start_date,end_date,"Power")
        sum_period2 = period_anal(start_date,end_date,"Total Electricity [GWh]")
        sum_period = [a/b for a, b in zip(sum_period1, sum_period2)]
    elif variable == "Renewable Energy Ratio [%]":
        sum_period1 = period_anal(start_date,end_date,"Total Renewable [GWh]")
        sum_period2 = period_anal(start_date,end_date,"Total Electricity [GWh]")
        sum_period = [a/b for a, b in zip(sum_period1, sum_period2)]
    else: 
        sum_period=period_anal(start_date,end_date,variable)
    fig=px.bar(x=countries, y=sum_period)
    fig.update_layout(title=f'{variable} for all countries in Europe',
                      xaxis_title='Countries',
                      yaxis_title=f'{variable}')
    return fig

@app.callback(Output('clicked-location2', 'children'),
        Output("sector_graph", "figure"),
        Output('pie-graph', 'figure'),
        Input("co2-map", "clickData"),
        Input("energy-map", "clickData"),
        Input("meteo-map", "clickData"),
        Input("sector_dropdown", "value"))

def data_analysis(click_data_co2,click_data_energy,click_data_meteo, sector):
    
    global click_data
    
    if click_data_co2:
        click_data = click_data_co2
    elif click_data_energy:
        click_data = click_data_energy
    elif click_data_meteo:
        click_data = click_data_meteo
    else:
        click_data = None
        return "No country selected", {}, {}
    
    selected_country = click_data['points'][0]['location']
    
    df=data_analysis_sing(selected_country)
    
    if sector == "CO2 Emissions" :
        df= df[['Total CO2 [Tonne]',
                        'Ground Transport',
                        'International Aviation',
                        'Residential',
                        'Industry',
                        'Domestic Aviation']]
        labels=list(df.iloc[:, 1:].columns)
        values=list(df.iloc[:, 1:].sum())
        
    elif sector=="Energy Consumption":
        df=df[['Other sources',
                        'Gas',
                        'Oil',
                        'Coal',
                        'Wind',
                        'Nuclear',
                        'Solar',
                        'Hydroelectricity',
                        'Total Renewable [GWh]',
                        'Total Non-Renewable [GWh]',
                        'Total Electricity [GWh]']]
        labels=list(df.iloc[:, :8].columns)
        values=list(df.iloc[:, :8].sum())
        
    elif sector=="Climate data":
        
        df=df[['Temperature [ºC]',
                            'Relative Humidity (%)',
                            'Rain [mm/h]',
                            'Wind Speed [km/h]',
                            'Pressure [mbar]',
                            'Solar Radiation [W/m^2]']]
        labels=[]
        values=[]
    
    fig2=px.line(df)
    
    fig3 = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                             insidetextorientation='radial'
                            )])

    
    if sector=="CO2 Emissions":
        fig2.update_layout(title=f'{sector} segregation for {selected_country}',
                        xaxis_title='Date',
                        yaxis_title='Values [Tonne]')
        fig3.update_layout(title=f'Contribution of each factor for {sector}')
        fig3.show()
    elif sector=="Energy Consumption":
        fig2.update_layout(title=f'{sector} segregation for {selected_country}',
                        xaxis_title='Date',
                        yaxis_title='Values [GWh]')
        fig3.update_layout(title=f'Contribution of each factor for {sector}')
        fig3.show()
    elif sector=="Climate data":
        fig2.update_layout(title=f'{sector} segregation for {selected_country}',
                        xaxis_title='Date',
                        yaxis_title='Values [Different Units]')
    
    return f"You are making the data analysis to {selected_country}.", fig2,fig3

@app.callback(Output('clicked-location3', 'children'),
        Output("correlation-graph", "figure"),
        Input("co2-map", "clickData"),
        Input("energy-map", "clickData"),
        Input("meteo-map", "clickData"),
        Input("correlation_dropdown", "value"))

def data_analysis_correlation(click_data_co2,click_data_energy,click_data_meteo,correlation):
    
    global click_data
    
    if click_data_co2:
        click_data = click_data_co2
    elif click_data_energy:
        click_data = click_data_energy
    elif click_data_meteo:
        click_data = click_data_meteo
    else:
        click_data = None
        return f"No country selected", {}
    
    selected_country = click_data['points'][0]['location']
    
    var1, var2, var1title, var2title = "", "", "", ""
    
    if correlation == "CO2 Emission and Temperature":
        var1='Temperature [ºC]'
        var2='Total CO2 [Tonne]'
        
        var1title= 'Temperature [ºC]'
        var2title= 'Total CO2 [Tonne]'
    
    elif correlation == "Energy Production and Temperature":
        var1='Temperature [ºC]'
        var2="Total Electricity [GWh]"
        
        var1title= 'Temperature [ºC]'
        var2title= "Total Electricity [GWh]"
        
    elif correlation == "CO2 Emission and Total Energy Produced":
        var1 = "Total Electricity [GWh]"
        var2 = "Power"
        
        var1title= "Fossil Fuel Energy Production [GWh]"
        var2title= "Energy Production CO2 Emissions [Tonne]"
    
    elif correlation == "Renewable Energy and Total Energy Produced":
        var1 = "Total Electricity [GWh]"
        var2 = "Total Renewable [GWh]"
        
        var1title= "Fossil Fuel Energy Production [GWh]"
        var2title= "Energy Production CO2 Emissions [GWh]"
        
    
    x,y,res = MetricForecast.relation(selected_country ,var1, var2)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data Relation'))
    fig.add_trace(go.Scatter(x=x, y=res.intercept + res.slope*x, mode='lines', name=f'Linear Regression'))

    fig.update_layout(
        title=f'{correlation} for {selected_country}',
        xaxis_title=var1title,
        yaxis_title=var2title
    )

    
    return f"You are making the data analysis to {selected_country}.", fig


@app.callback(Output('clicked-location4', 'children'),
        Output("features_bar", "figure"),
        Output("score_bar", "figure"),
        Input("co2-map", "clickData"),
        Input("energy-map", "clickData"),
        Input("meteo-map", "clickData"),
        Input("feature_dropdown", "value"),
        Input("fore_dropdown", "value"))

def feature_analysis(click_data_co2,click_data_energy,click_data_meteo, method, fore):
    
    global click_data
    
    if click_data_co2:
        click_data = click_data_co2
    elif click_data_energy:
        click_data = click_data_energy
    elif click_data_meteo:
        click_data = click_data_meteo
    else:
        click_data = None
        return f"No country selected", {}, {}
    
    selected_country = click_data['points'][0]['location']
    
    features_data = get_features(selected_country)

    if features_data is None:
        return f"No features data available for {selected_country}.", {}, {}

    [kbest_co, kbest_en, rfor_co, rfor_en, points_co_dic, points_en_dic] = features_data

    if fore == "CO2 Emission":
        dic2 = points_co_dic #_co
        
        if method == "kBest":
            dic1 = kbest_co
    
        elif method == "Random Forest Regressor":
            dic1 = rfor_co
        
    elif fore == "Renewable Energy":
        dic2 = points_en_dic #_en
        
        if method == "kBest":
            dic1 = kbest_en
    
        elif method == "Random Forest Regressor":
            dic1 = rfor_en
    
    #For dic1
    feature_names = list(dic1.keys())
    scores = [entry for entry in dic1.values()]

    #For dic2
    feature_names2 = list(dic2.keys())
    scores2 = [entry[1] for entry in dic2.values()]

    # Create the bar plots    
    fig1=px.bar(x=feature_names, y=scores)

    fig2=px.bar(x=feature_names2, y=scores2)
    
    
    fig1.update_layout(title=f'Features Score for {fore} using {method}',
                    xaxis_title='',
                    yaxis_title='Score')
    
    fig2.update_layout(title=f'Final Features Score for {fore}',
                    xaxis_title='',
                    yaxis_title='Score')
    
    
    
    return f"You are analysing Features for {selected_country}." , fig1, fig2


@app.callback([Output("forecast-graph", "figure"), Output("metrics-table", "children")],
              [Input("forecast-dropdown", "value"),
               Input("co2-map", "clickData"),
               Input("energy-map", "clickData"),
               Input("meteo-map", "clickData"),
               Input("metrics-checklist", "value")])
    
def update_forecast(selected_observable, click_data_co2, click_data_energy, click_data_meteo, selected_metrics):
    global click_data
    if click_data_co2:
        click_data = click_data_co2
    elif click_data_energy:
        click_data = click_data_energy
    elif click_data_meteo:
        click_data = click_data_meteo
    else:
        click_data = None
        return {}, html.Div()
        
    selected_country = click_data['points'][0]['location']
    df_plot, df_metrics = MetricForecast.Forecast(selected_country, selected_observable)
    df_plot.rename(columns={"y_pred": "Forecast"}, inplace=True)
    df_test = MetricForecast.df_test_dict[selected_country]
    if selected_observable == "CO2 Emissions":
        df_plot["Real Data"] = df_test["Total CO2 [Tonne]"]
    elif selected_observable == "Renewable Energy":
        df_plot["Real Data"] = df_test["Total Renewable [GWh]"]
    
    fig = px.line(df_plot, y=df_plot.columns[[0,1]])
    fig.update_yaxes(title_text=selected_observable)
    table = generate_table(df_metrics.loc[:,selected_metrics])
    return fig, table

@app.callback(
    Output('email-status', 'children'),
    [Input('send-button', 'n_clicks')],
    [State('email-body', 'value')]
)

def send_email(n_clicks, email_body):
    if n_clicks > 0:
        try:
            # Define your email parameters
            sender_email = 'forecast2019tecnico@gmail.com'  # Replace with your email
            receiver_email = 'forecast2019tecnico@gmail.com'  # Replace with recipient's email
            subject = 'Feedback'
            
            # Create the email message
            msg = MIMEText(email_body)
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = receiver_email
            
            # Connect to the SMTP server
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender_email, 'fjrf yzpn cknz xeiv')  # Replace with your email password
                server.sendmail(sender_email, receiver_email, msg.as_string())
        
            return html.Div('Email sent successfully!')
        
        except Exception as e:
            traceback.print_exc()  # Print traceback for debugging
            return html.Div(f'Error sending email: {str(e)}')


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)