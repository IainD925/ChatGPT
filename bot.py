import logging
import socket
import time
from ib_insync import IB, Contract, MarketOrder, LimitOrder, util
import datetime
import numpy
import pandas
import talib # P.SAR
import matplotlib.pyplot as plt  # charting
import matplotlib.dates as mdates

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# bot.disconnect()
# Define the connection and event handlers
class TradingBot:
    def __init__(self):
        self.ib = IB()
    
    def connect(self, host, port, client_id):
        self.ib.connect(host, port, clientId=client_id)
    
    def disconnect(self):
        self.ib.disconnect()
        
    def accountSummary(self):
        account_summary = self.ib.accountSummary()
        for summary in account_summary:
            print(summary.tag, summary.value)
    
    def get_crypto_contract(self, symbol, exchange='PAXOS', currency='USD'):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = 'CRYPTO'
        contract.exchange = exchange
        contract.currency = currency
        return contract
    
    def get_crypto_data(self, symbol):
        contract = self.get_crypto_contract(symbol)
        ticker = self.ib.reqMktData(contract, genericTickList='456', snapshot=True)
        self.ib.sleep(2)  # Give some time for the data to be received
        if ticker is None:
            print("Error: Ticker not available.")
            return None
        return ticker.last
    
    def get_stock_contract(self, symbol, exchange='SMART', currency='USD'):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = 'STK'
        contract.exchange = exchange
        contract.currency = currency
        return contract
    
    def get_stock_data(self, symbol, duration='1 D', bar_size='1 min'):
        contract = self.get_stock_contract(symbol)
        bars = self.ib.reqHistoricalData(contract, endDateTime='', durationStr=duration, barSizeSetting=bar_size, whatToShow='TRADES', useRTH=True)
        return bars
    
    def get_contract(self, symbol, contract_type):
        if contract_type == 'CRYPTO':
            return self.get_crypto_contract(symbol)
        elif contract_type == 'STOCK':
            return self.get_stock_contract(symbol)
        else:
            raise ValueError("Invalid contract type.")
    
    def create_order(self, action, quantity, order_type='MARKET', limit_price=None, tif=None):
        if order_type == 'MARKET':
            order = MarketOrder(action, quantity)
        elif order_type == 'LIMIT' and limit_price is not None:
            order = LimitOrder(action, quantity, limit_price)
            if tif:
                order.tif = tif
        else:
            raise ValueError("Invalid order type or missing limit price for limit order")
        return order
  
    def submit_order(self, symbol, order, contract_type='STOCK'):
        contract = self.get_contract(symbol, contract_type)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)  # Wait for order status to be updated
        return trade
       
    def get_historical_data(self, symbol, contract_type='STOCK', duration='30 D', bar_size='1 hour'):
        if contract_type == 'CRYPTO':
            contract = self.get_crypto_contract(symbol)
            what_to_show = 'AGGTRADES'
        else:
            contract = self.get_stock_contract(symbol)
            what_to_show = 'TRADES'
        historical_data = self.ib.reqHistoricalData(contract, endDateTime='', durationStr=f'{duration}', barSizeSetting=f'{bar_size}', whatToShow=what_to_show, useRTH=True)
        df = util.df(historical_data)
        df.index = pandas.to_datetime(df['date'])
        return df
    
    def get_parabolic_sar(self, symbol, contract_type='STOCK', duration='30 D', bar_size='1 hour'):
        if contract_type == 'CRYPTO':
            contract = self.get_crypto_contract(symbol)
            what_to_show = 'AGGTRADES'
        else:
            contract = self.get_stock_contract(symbol)
            what_to_show = 'TRADES'
        historical_data = self.ib.reqHistoricalData(contract, endDateTime='', durationStr=duration, barSizeSetting=bar_size, whatToShow=what_to_show, useRTH=True)
        df   = util.df(historical_data)
        high = numpy.array(df['high'], dtype=float)
        low  = numpy.array(df['low'], dtype=float)
        parabolic_sar = talib.SAR(high, low)
        return parabolic_sar
    
    def generate_signals(self, historical_data):
        df = historical_data.copy()
        high = numpy.array(df['high'], dtype=float)
        low = numpy.array(df['low'], dtype=float)
        parabolic_sar = talib.SAR(high, low)
        signals = []
        for i in range(1, len(parabolic_sar)):
            if parabolic_sar[i] < df.loc[df.index[i], 'close'] and parabolic_sar[i - 1] >= df.loc[df.index[i - 1], 'close']:
               signals.append(('BUY', df.index[i]))
            elif parabolic_sar[i] > df.loc[df.index[i], 'close'] and parabolic_sar[i - 1] <= df.loc[df.index[i - 1], 'close']:
                    signals.append(('SELL', df.index[i]))
        df['signal'] = 0
        for signal, date in signals:
            if signal == 'BUY':
                df.loc[date, 'signal'] = 1
            elif signal == 'SELL':
                df.loc[date, 'signal'] = -1
        return df
    
    def get_latest_SAR_signal_df(self, df, symbol):
        latest_buy_signal = None
        latest_sell_signal = None
        for i in reversed(df.index):
            signal = df.loc[i, 'signal']
            if signal == 1 and latest_buy_signal is None:
                latest_buy_signal = i
            elif signal == -1 and latest_sell_signal is None:
                latest_sell_signal = i
            # Break the loop if both latest buy and sell signals are found
            if latest_buy_signal and latest_sell_signal:
                break
        data = {
            'symbol': [symbol, symbol],
            'signal': ['BUY', 'SELL'],
            'date': [latest_buy_signal, latest_sell_signal]
        }
        signal_df = pandas.DataFrame(data)
        return signal_df

def plot_chart(df,chart_ticker):
        # Convert the index (dates) to datetime objects
        df.index = pandas.to_datetime(df.index, unit='s')
        
        # Prepare the figure and axis
        fig, ax = plt.subplots(figsize=(15, 10))
        # Plot the close price
        ax.plot(df.index, df['close'], label='Close Price', color='blue', linewidth=2)
        # Plot the Parabolic SAR buy signals (where signal == 1)
        ax.scatter(df[df['signal'] == 1].index, df[df['signal'] == 1]['close'], label='Buy Signal', marker='^', color='g', s=100)
        # Plot the Parabolic SAR sell signals (where signal == -1)
        ax.scatter(df[df['signal'] == -1].index, df[df['signal'] == -1]['close'], label='Sell Signal', marker='v', color='r', s=100)
        # Set the title and labels
        ax.set_title(f"{chart_ticker['symbol']} Price with Parabolic SAR Buy and Sell Signals")
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        # Set the date format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        # Show the legend
        ax.legend(loc='upper left')
        # Display the chart
        plt.show()

# Define backtesting function
def backtest_strategy(df, initial_cash=100000, commission=0.0015):
    cash = initial_cash
    position = 0
    trades = []
    
    for i, row in df.iterrows():
        if row['signal'] == 1:  # Buy signal
            if position == 0:  # If no current position, buy
                position_size = cash / row['close']  # Calculate position size
                position = position_size  # Update position
                cash -= position_size * row['close'] * (1 + commission)  # Update cash
                
                trades.append(('BUY', i, row['close'], position_size))
                
        elif row['signal'] == -1:  # Sell signal
            if position > 0:  # If holding a position, sell
                cash += position * row['close'] * (1 - commission)  # Update cash
                trades.append(('SELL', i, row['close'], position))
                
                position = 0  # Reset position
                
    # Calculate final portfolio value
    portfolio_value = cash + position * df.iloc[-1]['close']
    return portfolio_value, trades

# End define bot Functions #

# Initialize the trading bot
bot = TradingBot()

# Connect to Interactive Brokers API
# bot.disconnect()
bot.connect(host='127.0.0.1', port=7497, client_id=1)  # Live port: 7496 Test: 7497

# Check connection / Call accountSummary
# bot.accountSummary()

## BTC
# Check BTC Price
btc_price = bot.get_crypto_data('BTC')
if btc_price: print(f"BTC Price: {btc_price}")
else: print("BTC Price not available")

# Get historical data for Bitcoin
historical_data = bot.get_historical_data('BTC', contract_type='CRYPTO', duration='90 D', bar_size='1 hour')

# Generate signals
signal_df = bot.generate_signals(historical_data)

# Return latest BUY & SELL SAR signals
signal_SAR_df = bot.get_latest_SAR_signal_df(signal_df, 'BTC')
print(signal_SAR_df)

# Return just the latest signal
latest_signal = signal_SAR_df.iloc[-1].to_dict()
print(latest_signal)

# Plot the chart
plot_chart(signal_df,latest_signal)

# ! TRADING ! #
if latest_signal['signal'] == 'BUY':
    # Buy 0.001 BTC. Note: IB doesn't support GTC for Crypto. Options are Immediate Or Cancel (IOC) or Minutes (cancels after 5 minutes)
    limit_buy_order = bot.create_order('BUY', 0.001, order_type='LIMIT', limit_price=btc_price, tif='IOC')
    trade = bot.submit_order('BTC', limit_buy_order, contract_type='CRYPTO')
    print(f"Buy 0.001 BTC at USD: {trade}")
elif latest_signal['signal'] == 'SELL':
    # Sell 0.001 BTC
    limit_sell_order = bot.create_order('SELL', 0.001, order_type='LIMIT', limit_price=btc_price, tif='IOC')
    trade = bot.submit_order('BTC', limit_sell_order, contract_type='CRYPTO')#
    print(f"Sell 0.001 BTC at 33000 USD: {trade}")
else:
    print("No signal or unrecognized signal.")

# STOCKS
# Request historical stock data for QQQ
stock_historical_data = bot.get_historical_data('QQQ', contract_type='STOCK', duration='90 D', bar_size='1 hour')
# Generate signals
stock_signal_df = bot.generate_signals(stock_historical_data)

# Return latest BUY & SELL SAR signals
stock_signal_SAR_df = bot.get_latest_SAR_signal_df(stock_signal_df, 'QQQ')
print(stock_signal_SAR_df)

# Return just the latest signal
stock_latest_signal = stock_signal_SAR_df.iloc[-1].to_dict()
print(stock_latest_signal)

# Plot the chart
plot_chart(stock_signal_df,stock_latest_signal)

### BACKTESTING
# Run backtest on signal_df
initial_value = 10000
def run_backtest(symbol, signal_df):
    final_value, trades = backtest_strategy(signal_df,initial_value)
    
    # Calculate performance metrics
    return_percentage = ((final_value - initial_value) / initial_value) * 100
    print(f"{symbol} Backtesting Results:")
    print(f"Initial value: {initial_value}")
    print(f"Final value: {final_value}")
    print(f"Return: {return_percentage}%")
    print(f"Number of trades: {len(trades)}")
    print()

# Run backtests for BTC and QQQ
run_backtest('BTC', signal_df)
run_backtest('QQQ', stock_signal_df)

# Start the TCP server to listen for buy/sell instructions
# bot.start_server(host='localhost', port=9999)
# bot.disconnect()