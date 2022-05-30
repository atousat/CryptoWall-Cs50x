''' --------- Import Libraries  ---------  '''
''' ************************************************'''

# Import the Libraries
import pandas as pd
import numpy as np

# For time stamps
from datetime import datetime

# To draw a diagram
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
#%matplotlib inline

# To draw a hit map
import seaborn as sns
sns.set_style('whitegrid')

# For Get API from coinmarketcap.com
import time
import json
from bs4 import BeautifulSoup
from requests import Request, Session

# To get information from Yahoo
import yfinance as yf 

# For the LSTM
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM



''' --------- Download stock data from yahoo then export as CSV ---------  '''
''' ************************************************************************'''

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# Download stock data then export as CSV
df_btc = yf.download("BTC-USD", start, end)
df_btc.to_csv('bitcoin.csv')

df_eth = yf.download("ETH-USD", start, end)
df_eth.to_csv('ethereum.csv')

df_usdt = yf.download("USDT-USD", start, end)
df_usdt.to_csv('tether.csv')

df_ada = yf.download("USDC-USD", start, end)
df_ada.to_csv('usdc-coin.csv')

df_bnb = yf.download("BNB-USD", start, end)
df_bnb.to_csv('binance-coin.csv')


''' --------- Analyze the closing prices from dataset: ---------  '''
''' ************************************************************************'''

# Plot stock company['Close']
def figure_close(stockname , name):
    df = pd.read_csv(stockname)
    df["Date"] = pd.to_datetime(df.Date,format="%Y-%m-%d")
    df.index = df['Date']
    
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot()
    ax.set_title(name)
    ax.set_xlabel('Date', fontsize=16)
    ax.set_ylabel('Close Price USD ($)', fontsize=16)
    plt.plot(df["Close"],label='Close Price history')


''' ------- Use sebron for a quick correlation plot for the daily returns -------  '''
''' *********************************************************************************'''

# The tech stocks we'll use for this analysis
coin_list = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'USDC-USD', 'BNB-USD']

# Download stock data then export as CSV
df_close = yf.download(coin_list, start, end)['Adj Close']
df_close.to_csv('closing.csv')

# Plot all the close prices
((df_close.pct_change()+1).cumprod()).plot(figsize=(16,8))

# Show the legend
plt.legend()

# Define the label for the title of the figure
plt.title("Returns", fontsize=16)
plt.ylabel('Cumulative Returns', fontsize=14)
plt.xlabel('Year', fontsize=14)

# Plot the grid lines
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()

# Haetmap
sns.heatmap(df_close.corr(), annot=True, cmap='summer')


''' ------- Get API from coinmarketcap.com -------  '''
''' ************************************************************'''

url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
  'start':'1',
  'limit':'100',
  'convert':'USD'
}
headers = {
  'Accepts': 'application/json',
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

# Convert json data to CSV file
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


''' ------- Get dataset from yfinance for historical data of cryptocurrencies -------  '''
''' ************************************************************************************'''

# create empty dataframe
coin_final = pd.DataFrame()
Symbols = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'USDC-USD', 'BNB-USD']

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

for i in Symbols:   
    # print the symbol which is being downloaded
    print( str(Symbols.index(i)) + str(' : ') + i, sep=',', end=',', flush=True)  
    
    try:
        # download the stock price 
        coin = []
        coin = yf.download(i,start=start, end=end, progress=False)
        
        # append the individual stock prices 
        if len(coin) == 0:
            None
        else:
            coin['Name']=i
            coin_final = coin_final.append(coin,sort=False)
    except Exception:
        None
# convert to csv file        
coin_final.to_csv('5coins.csv')



''' ------- Predicting the closing crypto price with LSTM -------  '''
''' *********************************************************************'''

# For Warning 
pd.options.mode.chained_assignment = None  # default='warn'

def LSTM_Model(stockname):
    # read the stock Price
    df = pd.read_csv(stockname)
    df["Date"] = pd.to_datetime(df.Date,format="%Y-%m-%d")
    df.index = df['Date']
    
    # Create a new dataframe with only the 'Close' column
    data = df.filter(['Close'])
    
    # Converting the dataframe to a numpy array
    dataset = data.values
    
    # Get /Compute the number of rows to train the model on
    training_data_len = math.ceil(len(dataset) *.8)
    
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
    print("rmse: ",rmse)
    
    # Plot/Create the data for the graph
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    
    #Visualize the data
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()




''' ------- Print the result Function -------  '''
''' *****************************************************'''

LSTM_Model('bitcoin.csv')
figure_close('bitcoin.csv' , 'Bitcoin')

''' For other coins, 
    the output is as above, 
    just select the name of the coin
    (Ethereum/ Tether/ Cardano/ Binance-coin) 
    The names of the csv files:
    (ethereum.csv, tether.csv, cardano.csv, binance-coin.csv)''' 
