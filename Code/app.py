''' --------- Import Libraries ---------  '''
''' ************************************************'''

#  For Dashboard
import dash
import dash_auth
from dash_bootstrap_components._components.CardImg import CardImg
from dash_bootstrap_components._components.Row import Row
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Output, Input, State

# for Plot 
import plotly.express as px
import plotly.graph_objs as go

# use boostrap
import dash_bootstrap_components as dbc   

# For read the dataset 
import pandas as pd

# For mathematical operations 
import numpy as np   
 
# To get information from Yahoo
import yfinance as yf 
from yahoo_fin import stock_info as si #To get an instant price

# For time stamps
from datetime import datetime

# For Get API from coinmarketcap.com
import json
from requests import Request, Session

# For the LSTM
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM



''' --------- Get API from coinmarketcap.com ---------  '''
''' ************************************************'''
url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
  'start':'1',
  'limit':'100',
  'convert':'USD'
}
headers = {
  'Accepts': 'application/json',
  # My API Key
  'X-CMC_PRO_API_KEY': 'd50d92c4-c400-4e06-9ab7-b630f977e94c',
}

session = Session()
session.headers.update(headers)

response = session.get(url, params=parameters)
json = json.loads(response.text)
data = json['data']

coins ={}
for x in data:
    coins[str(x['id'])] = x['slug']

coin_name = []
coin_symbol = []
market_cap = []
percent_change_1h = []
percent_change_24h = []
percent_change_7d = []
price = []
volume_24h = []

for i in data:
    coin_name.append(i['slug'])
    coin_symbol.append(i['symbol'])
    price.append(i['quote']['USD']['price'])
    percent_change_1h.append(i['quote']['USD']['percent_change_1h'])
    percent_change_24h.append(i['quote']['USD']['percent_change_24h'])
    percent_change_7d.append(i['quote']['USD']['percent_change_7d'])
    market_cap.append(i['quote']['USD']['market_cap'])
    volume_24h.append(i['quote']['USD']['volume_24h'])

df = pd.DataFrame(columns=['coin_name', 'coin_symbol', 'market_cap', 'percent_change_1h', 'percent_change_24h', 'percent_change_7d', 'price', 'volume_24h'])
df['coin_name'] = coin_name
df['coin_symbol'] = coin_symbol
df['price'] = price
df['percent_change_1h'] = percent_change_1h
df['percent_change_24h'] = percent_change_24h
df['percent_change_7d'] = percent_change_7d
df['market_cap'] = market_cap
df['volume_24h'] = volume_24h

df.to_csv( "cryptocurrencies.csv", index=False, encoding='utf-8-sig')

# Read Dataset
dfs = pd.read_csv('cryptocurrencies.csv')




''' --------- Real-Time Crypto Price ---------  '''
''' ************************************************'''
# import stock_info module from yahoo_fin
def get_Price(coin):
    price = si.get_live_price(coin)
    return price




''' --------- Get dataset from yfinance ---------  '''
''' ************************************************'''
# create empty dataframe
coin_final = pd.DataFrame()
Symbols = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'ADA-USD', 'BNB-USD']

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

for i in Symbols:   
    # print the symbol which is being downloaded
    print( str(Symbols.index(i)) + str(' : ') + i, sep = ',', end = ',', flush = True)  
    
    try:
        # download the coin price 
        coin = []
        coin = yf.download(i,start = start, end = end, progress = False)
        
        # append the individual coin prices 
        if len(coin) == 0:
            None
        else:
            coin['Name']=i
            coin_final = coin_final.append(coin,sort=False)
    except Exception:
        None
# convert to csv file        
coin_final.to_csv('5coins.csv')

# Read dataset
df = pd.read_csv('5coins.csv')





''' --------- App ---------  '''
''' ************************************************'''
# username and password for admin singup
VALID_USERNAME_PASSWORD_PAIRS = {
    'atousa': '123456'
    }

# theme from https://www.bootstrapcdn.com/bootswatch/
app = dash.Dash(
        # responsive layout
        meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}],

        # bootstrap
        external_stylesheets=[dbc.themes.LUX])
app.title = "Cryptowall"   # title of App

# Basic Authentication
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)



'''---------- LSTM ----------'''
''' ************************************************'''

# read the stock Price
# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# Download stock data then export as CSV
df_btc = yf.download("BTC-USD", start, end)
df_btc.to_csv('bitcoin.csv')
df_btc = pd.read_csv('bitcoin.csv')
df_btc["Date"] = pd.to_datetime(df_btc.Date,format="%Y-%m-%d")
df_btc.index = df_btc['Date']
    
# Create a new dataframe with only the 'Close' column
data = df_btc.filter(['Close'])
    
# Converting the dataframe to a numpy array
dataset = data.values
    
# Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(dataset) *.8)
    
# Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
    
# Create the scaled training data set
train_data = scaled_data[0:training_data_len  , : ]
    
# Split the data into x_train and y_train data sets
x_train = []
y_train = []
    
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
        
# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
    
# Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    
# Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
    
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
    
# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)
    
# Test data set
test_data = scaled_data[training_data_len - 60: , : ]
    
# Create the x_test and y_test data sets
x_test = []
       
y_test =  dataset[training_data_len : , : ] 
    
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
        
# Convert x_test to a numpy array 
x_test = np.array(x_test)
    
# Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    
# Getting the models predicted price values
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)#Undo scaling
    
# Calculate/Get the value of RMSE
rmse = np.sqrt(np.mean(((predictions- y_test)**2)))
#rmse
    
# Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions


''' --------- Layout ---------  '''
''' ************************************************'''

app.layout = dbc.Container([
    
    # Title 
    dbc.Row([
        dbc.Col([
            html.H1('Cryptowall  🗠'),
            html.H6('CS50x Final Rroject')
        ], className='text-center text-dark, my-4')
    ], style = {"background-color": "#f9f9f9"}),

    # Tabs
    dcc.Tabs(id="tabs", children=[

        # ************************************* Tab-01 *************************************
        dcc.Tab(label='Top 100 Coins',children=[
            # Row-00
            dbc.Row([
                dbc.Col([
                    html.H1('top 100 cryptocurrency 2021 by Market Capitalization'),
                    dbc.Label('Get API from:     '),
                    dcc.Link('coinmarketcap.com', href='https://coinmarketcap.com'),
                ], className='text-center text-dark, my-4')
            ]),

            # Row-01('Download dataset')
            dbc.Row([
                dbc.Col([
                    # Export Datatable to csv file
                    html.Button("Download Dataset", id="btn_csv", className ="btn btn-outline-dark"), 
                    dcc.Download(id="download-csv"), 
                ])
            ]),
            html.Br(),
  
            # Row-02('Table')
            dbc.Row([
                dbc.Col([
                    dt.DataTable(id='table',
                        columns = [{"name": i, "id": i, 'deletable': True, "selectable": True} 
                                   for i in dfs.columns],
                        data = dfs.to_dict('records'),
                        fixed_rows = {'headers': True},
                        row_deletable = True,
                        row_selectable="multi",
                        page_size = 20, 

                        # Style of Table 
                        style_table = {'height': '300px','overflowY': 'auto', 'padding':'5px'},
                        style_header = {'minWidth': '40px', 'width': '180px', 'maxWidth': '200px',
                                        'whiteSpace': 'normal', 'height': 'auto', 'overflowY': 'auto'},
                        style_cell_conditional=[{'textAlign': 'left', 'padding': '5px'}],
                        style_data_conditional =[
                        {'if':{
                                'filter_query': 
                                '{percent_change_1h} < 0',
                                'column_id': 'percent_change_1h'
                            },
                            'color': 'tomato' 
                        }
                        ,
                        {'if':{
                                'filter_query': 
                                '{percent_change_24h} < 0',
                                'column_id': 'percent_change_24h'
                            },
                            'color': 'tomato' 
                        },
                        {'if':{
                                'filter_query': 
                                '{percent_change_7d} < 0',
                                'column_id': 'percent_change_7d'
                            },
                            'color': 'tomato' 
                        }
                        ]
                    )
                ]) 
            ]),

            # Row-03('Bar chart')
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.H1('Change price',className='text-center text-dark, my-4'),
                    html.Div(id='bar-chart')
                ])
            ],style = {"width": "85%" , "margin": "auto" })
        
        ]),
         
        # ************************************* Tab-02 *************************************
        dcc.Tab(label = 'Top 5 Coins' ,children=[

            # Row-00('Title')
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.H1("5 Top Coins by Market Cap "),
                    html.H5('Bitcoin, Ethereum, Tether, Cardano	, Binance-coin'),
                    html.P('Cryptocurrency Price, Chart, Histogram ')
                ], className='text-center text-dark, my-4')
            ]),
            
            # Row-01(Title for live price)
            dbc.Row([
                dbc.Col(
                    html.H1('Live Cryptocurrency Prices (USD)$', className='text-center text-dark, my-4')
                )
            ]),

            # Row-02 ('Data & Time')
            dbc.Row([
                dbc.Col([
                    html.H4(
                     datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
                ], className = 'text-center text-secondary')
            ]),

            # Row-03('Live Price')
            dbc.Row([
                # Col-00('bitcoin')
                dbc.Col(
                    dcc.Graph(
                        figure = {
                            'data':[
                                go.Indicator(
                                    title= {'text': "Bitcoin"},
                                    mode = "number",
                                    value = get_Price('BTC-USD')
                                    )],

                            "layout":
                                go.Layout(autosize = True, margin = {'t':0,'l':5,'b':0,'r':5})
                            })
                ),
                
                # Col-01('ethereum')
                dbc.Col(
                    dcc.Graph(
                        figure = {
                            'data':[
                                go.Indicator(
                                    title= {'text': "Ethereum"},
                                    mode = "number",
                                    value = get_Price('ETH-USD')
                                    )],

                            "layout":
                                go.Layout(autosize = True, margin = {'t':0,'l':5,'b':0,'r':5})
                            })
                ),
                
                # Col-02('tether')
                dbc.Col(
                    dcc.Graph(
                        figure = {
                            'data':[
                                go.Indicator(
                                    title= {'text': "Tether"},
                                    mode = "number",
                                    value = get_Price('USDT-USD')
                                    )],

                            "layout":
                                go.Layout(autosize = True, margin = {'t':0,'l':5,'b':0,'r':5})
                            })
                ),
                
                # Col-03('cardano')
                dbc.Col(
                    dcc.Graph(
                        figure = {
                            'data':[
                                go.Indicator(
                                    title= {'text': "cardano"},
                                    mode = "number",
                                    value = get_Price('ADA-USD')
                                    )],

                            "layout":
                                go.Layout(autosize = True, margin = {'t':0,'l':5,'b':0,'r':5})
                            })
                ),
                
                # Col-04('binance-coin')
                dbc.Col(
                    dcc.Graph(
                        figure = {
                            'data':[
                                go.Indicator(
                                    title= {'text': "Binance-coin"},
                                    mode = "number",
                                    value = get_Price('BNB-USD')
                                    )],

                             "layout":
                                go.Layout(autosize = True, margin = {'t':0,'l':5,'b':0,'r':5})
                            })           
                ),
                
            ]),
 
            # Row-04('Title for chart')
            dbc.Row([
                dbc.Col([
                    html.H1("Close Price USD ($) "  ,className='text-center text-dark, my-4')
                ])
            ]),

            # Row-05('DropDown')
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dcc.Dropdown(id = 'my-dpdn', multi = True , value = '',
                                options = [{'label':x, 'value':x}
                                for x in sorted(df['Name'].unique())
                        ], style = {"width": "95%" , "margin": "auto" })

                    ], style = {"width": "85%", "height":"6rem", "margin": "auto" , "padding": "10px",
                                 "box-shadow": "0 4px 6px 0 rgba(0, 0, 0, 0.18)"})  
                )
            ]),

            # Row-06('Chart')
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id = 'fig', figure = {"layout": {"title": "Close Prise"}}
                       # style
                       , className = "my-4",
                         style = {"box-shadow": "0 4px 6px 0 rgba(0, 0, 0, 0.15)",
                                 "width": "85%", "margin": "auto"}
                    )
                )
            ]),


            html.Br(), html.Br(),
            # Row-07('Title for histogram')
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.H1("Histogram Chart",className='text-center text-dark, my-4')
                ])
            ]),

            # Row-08('Histogram')
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            # title
                            html.H6("Select the Coins:", className = "font-weight-bolder"),

                            # Checklist(select the coin name)
                            dcc.Checklist(id = 'my-chek', value = '', options=[{'label':x, 'value':x}
                                         for x in sorted(df['Name'].unique())]
                                         , labelClassName="mr-4"),

                            # Chart
                            dcc.Graph(id='my-hist', figure={})
                            ])

                    ], style = {"width": "85%", "margin": "auto" ,   # style css
                               "box-shadow": "0 4px 6px 0 rgba(0, 0, 0, 0.18)"}),
                    html.Br(),html.Br(),
                ])
            ])

        ]), # end tab-02

        # ************************************* Tab-03 *************************************
        dcc.Tab(label='Forecast',children=[
            html.Br(),
            #Row-00('Coin-Name')
            dbc.Row([
                dbc.Col([
                    html.H1(' Bitcoin ₿', 
                    className = 'text-warning'),
                    html.H6('more information about Bitcoin: '),
                    dcc.Link('Bitcoin-Wikipedia', href='https://en.wikipedia.org/wiki/Bitcoin'),

                   dcc.Graph(id = 'fig-btc', 
                            figure = {
                                "data":[
                                    go.Scatter(
                                        x = df_btc['Date'],
                                        y = df_btc['Close'],
                                        mode = 'lines',
                                        line = dict(shape = 'linear', color = '#FFA500', dash = 'dot'),
                                        connectgaps = True
                                        )],

                                "layout": {"title": "Historical price in the past year"}
                                },
                            style = {"width": "50%", "margin": "auto"}
                            )
                ],className='text-center text-dark, my-4')
            ]),

            # Row-01 ( Close figure )
            dbc.Row([
                dbc.Col([
                    html.Br(),html.Br(),
                    # title 
                    html.H2("Close figure", className = 'text-center blockquote, my-4'),

                    # Chart 
                    dcc.Graph(id = "Close",
                    figure= { 
                        "data":[
                            go.Scatter(
                                x = train.index,
                                y = valid["Close"],
                                mode = 'lines',
                                line = dict(shape = 'linear', color = 'rgb(10, 120, 24)', dash = 'dot'),
                                connectgaps = True
                        )],
                        "layout":go.Layout(
                            title = 'scatter plot',
                            xaxis = {'title':'Date'},
                            yaxis = {'title':'Closing Rate'}
                        )}
                        
                        # style
                        , className = "my-4",
                        style = {"box-shadow": "0 4px 6px 0 rgba(0, 0, 0, 0.15)",
                                  "width": "85%", "margin": "auto"}
                       ),
                ])
            ]),
            html.Br(),html.Br(),
            
            # Row-02 ( 'predict Chart' )
            dbc.Row([
                dbc.Col([ 
                    #  Header and Description
                     html.H2("Predicted figure", className = 'text-center blockquote, my-4'),
                     html.P("This program uses an artificial recurrent neural network called"
                            " Long Short Term Memory (LSTM)", 
                              className="text-decoration-none text-center"),
                    
                    # Chart
                    dcc.Graph(id = "Predicted",
                    figure={
                        "data":[
                            go.Scatter(
                                x=valid.index,
                                y=valid["Predictions"],
                                mode = 'lines',
                                line = dict(shape = 'linear', color = 'rgb(100, 10, 100)', dash = 'dot'),
                                connectgaps = True
                            )],

                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )}
                        , className = "my-4",

                        # style code css
                        style = {"box-shadow": "0 4px 6px 0 rgba(0, 0, 0, 0.15)",
                                  "width": "85%", "margin": "auto"}),
                    html.Br(), html.Br()
                ])
            ])
        ])#end tab-03

    ]) # end tabs


], fluid= True)

''' --------- Callback ---------  '''
# --------------- Output Tab-01 ---------------
# Download csv file
@app.callback(
    Output("download-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True
    )

def func(n_clicks):
    return dcc.send_data_frame(dfs.to_csv, "Data.csv")


# Show Bar chart
@app.callback(
    Output('bar-chart', "children"),
    Input('table', "derived_virtual_data"),
    Input('table', "derived_virtual_selected_rows")
    )

def update_graphs(rows, derived_virtual_selected_rows):
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    dffs = dfs if rows is None else pd.DataFrame(rows)

    colors = ['#7FDBFF' if i in derived_virtual_selected_rows else '#0074D9'
              for i in range(len(dffs))]

    return [
        dcc.Graph(
            id = column,
            figure = {
                "data": [
                    {
                        "x": dffs["coin_name"],
                        "y": dffs[column],
                        "type": "bar",
                        "marker": {"color": colors},
                    }
                ],
                "layout": {
                    "xaxis": {"automargin": True},
                    "yaxis": {
                        "automargin": True,
                        "title": {"text": column}
                    },
                    "height": 250,
                    "margin": {"t": 10, "l": 10, "r": 10},
                },
            },
        )
        
        for column in ["percent_change_24h", "percent_change_7d"] if column in dffs]


# --------------- Output Tab-02 ---------------
# Show Chart
@app.callback(
    Output('fig', 'figure'),
    Input('my-dpdn', 'value')
    )

def close_graph(coin_slctd):
    dff = df[df['Name'].isin(coin_slctd)]
    figln = px.line(dff, x = 'Date', y = 'Close', color = 'Name')
    return figln

# Show Histogram
@app.callback(
    Output('my-hist', 'figure'),
    Input('my-chek', 'value')
    )

def update_graph(coin_slctd):
    dff = df[df['Name'].isin(coin_slctd)]
    dff = dff[dff['Date']=='2021-10-05']  # Date Today
    fighist = px.histogram(dff, x = 'Name', y = 'Close', color = 'Name')
    return fighist



''' --------- Run Server ---------  '''
''' ************************************************'''
if __name__ == "__main__":
    app.run_server()