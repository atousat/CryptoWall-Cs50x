# CryptoWall
### Video Demo: https://youtu.be/R0aJ394fE2k
### Description:
In this project, we will build a website in the field of cryptocurrency that displays the information of the top 100 coins based on the Coinmarketcap site in the form of a table
and their price changes in the form of a chart.
In the following, we will analyze the top 5 coins and then go to the Bitcoin price forecast by the LSTM algorithm.
We display the results of this research in the form of a dashboard as a website.

The dashboard consists of three main parts, which include:
### 1. Top 100 Coins:
> in this part:
- The top 100 coins with their specifications are displayed in a table. This profile is obtained by getting the API from coinmarketcap.com.
- It is possible to download the database as a Csv file.
- Price changes in the last twenty-four hours and the last seven days are displayed as a Chart Bar.

### 2. Top 5 Coins:
> In this section, we review the top five coins, which include:
- Show instant price and price change.
- Price change chart in the past year based on Close.
- Histogram

### 3. Forecast:
> In this section, we will forecast the stock price of Bitcoin using LSTM.
- Memory Term-Short Long is an artificial repetitive neural network (RNN) architecture used for deep learning.

** It is possible to download all charts as photos. **

The following items have been used to implement this project:

## Front End Implementation:
The Dash library in Python has been used to implement and design the site pages.Dash is based on Flask, React.js and plotly.js.
Html tags are used to implement Dash, and Css is used to style pages.

## Back end implementation:
Python programming language has been used to implement the site.

- The site runs on Heroku server for free.
