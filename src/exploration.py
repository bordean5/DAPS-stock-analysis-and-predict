""" the module is for the explore part of the daps-final

the module explore the data with common plot, correlations, scatter plot...

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

def explore_plot(stock_frame, oil_frame):
    """
    the whole exploration procedure 
    Args:
      stock_frame: stock data after preprocessing
      oil_frame: WTI data after preprocessing
    Returns:
      plots in folder
    """
    #merge the stock and oil data with date for plot
    close=stock_frame[['date','close','log_volume']]
    oil=oil_frame[['date','value']]
    merge_frame = pd.merge(close,oil,on='date',how='left')
    merge_frame.rename(columns={'close': 'stock_price'}, inplace=True)
    merge_frame.rename(columns={'value': 'WTI'}, inplace=True)
    merge_frame.set_index('date',inplace=True)

    # Calculating and plotting moving averages with different time windows 
    ma_30= merge_frame['stock_price'].rolling(window=30).mean()
    ma_90 = merge_frame['stock_price'].rolling(window=90).mean()
    #plot the stock price with 30/90 moving average
    plt.figure(figsize=(14, 6))
    plt.plot(merge_frame['stock_price'], label='AAL Stock Price', color='blue', alpha=0.5)
    plt.plot(ma_30, label='30-Day Moving Average', color='red')
    plt.plot(ma_90, label='90-Day Moving Average', color='green')
    plt.title('AAL Stock Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig('save_image/explore/Stock price with MAs.png')
    # plot the seasonality
    plt.clf()
    decompose_result = seasonal_decompose(merge_frame['stock_price'], model='additive', period=365)
    decompose_result.plot()
    plt.tight_layout()
    plt.savefig('save_image/explore/Stock price seasonality')

    # calculate and plot Correlation matrix
    plt.clf()
    correlation_matrix = merge_frame.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('save_image/explore/Correlation Matrix')
  
    #plot scatter plot of log_volume and stock price
    plt.clf()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=merge_frame['log_volume'], y=merge_frame['stock_price'])
    plt.title('Scatter Plot of AAL Stock Price vs volume(log)')
    plt.xlabel('volume(log)')
    plt.ylabel('AAL Stock Price')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('save_image/explore/Scatter Plot of AAL Stock Price vs volume(log)')
  
    #plot scatter plot of lWTI and stock price
    plt.clf()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=merge_frame['WTI'], y=merge_frame['stock_price'])
    plt.title('Scatter Plot of AAL Stock Price vs WTI Value')
    plt.xlabel('Value (WTI Price)')
    plt.ylabel('AAL Stock Price')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('save_image/explore/Scatter Plot of AAL Stock Price vs WTI')

    pre_pandemic_data = merge_frame[merge_frame.index < '2021-05-01']
  
    #plot scatter plot of lWTI and stock price before 2021-5
    plt.clf()
    plt.figure(figsize=(14, 6))

    # AAL Stock Price (Pre-Pandemic)
    plt.subplot(2, 1, 1)
    plt.plot(pre_pandemic_data['stock_price'], label='AAL Stock Price (Pre-Pandemic)', color='blue')
    plt.title('AAL Stock Price Over Time (Pre-Pandemic)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

   # WTI Price (Pre-Pandemic)
    plt.subplot(2, 1, 2)
    plt.plot(pre_pandemic_data['WTI'], label='WTI Price (Pre-Pandemic)', color='green')
    plt.title('WTI Price Over Time (Pre-Pandemic)')
    plt.xlabel('Date')
    plt.ylabel('WTI Price')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('save_image/explore/AAL stock price and WTI before 2021-5-1' )

    # Recalculating the correlation coefficient for the pre-pandemic period
    pre_pandemic_correlation_coefficient = pre_pandemic_data['stock_price'].corr(pre_pandemic_data['WTI'])
    pre_pandemic_output='The correlation between stock price and WTI before 2021-5: '+str(pre_pandemic_correlation_coefficient)
    print(pre_pandemic_output)
  
    #plot scatter plot of lWTI and stock price after 2021-5
    post_pandemic_data = merge_frame.loc['2021-05-01':]

    # Plotting pre-pandemic trends
    plt.clf()
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    plt.plot(post_pandemic_data['stock_price'], label='AAL Stock Price (Post-Pandemic)', color='blue')
    plt.title('AAL Stock Price Over Time (Post-Pandemic)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(post_pandemic_data['WTI'], label='WTI Price (Post-Pandemic)', color='green')
    plt.title('WTI Price Over Time (Post-Pandemic)')
    plt.xlabel('Date')
    plt.ylabel('WTI Price')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('save_image/explore/AAL stock price and WTI after 2021-5-1' )

    # Recalculating the correlation coefficient after 2021-5
    post_pandemic_correlation_coefficient = post_pandemic_data['stock_price'].corr(post_pandemic_data['WTI'])
    post_pandemic_output='The correlation between stock price and WTI after 2021-5: '+str(post_pandemic_correlation_coefficient)
    print(post_pandemic_output)

    plt.clf()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pre_pandemic_data['WTI'], y=pre_pandemic_data['stock_price'])
    plt.title('Scatter Plot of AAL Stock Price vs Value before 2020-5')
    plt.xlabel('Value (WTI Price/Sales Data)')
    plt.ylabel('AAL Stock Price')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('save_image/explore/Scatter Plot of AAL Stock Price vs WTI before 2020-5')


def indicators_calculate_plot(stock_frame):
    """
    calculate and plot the indicators
    Args:
      stock_frame: stock data after preprocessing
    Returns:
      plots in folder
    """
    close=stock_frame[['date','high','low','open','close','volume']].copy()
    close.set_index('date',inplace=True)
    # Function to calculate Exponential Moving Average 
    def calculate_ema(prices, days):
        price=prices.copy()
        return price.ewm(span=days, adjust=False).mean()

    # Calculate MACD default setting(12,26,9)
    short_ema = calculate_ema(close.loc[:,'close'], 12)
    long_ema = calculate_ema(close.loc[:,'close'], 26)
    close['macd'] = short_ema - long_ema
    close['macd_signal'] = calculate_ema(close['macd'], 9)
    close.loc[:,'macd_hist'] = close['macd'] - close['macd_signal']

    # Calculate RSI with common 14 time period
    delta = close['close'].diff()  #measure the differnece between the one and last one
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    close['rsi'] = 100 - (100 / (1 + rs))

    # Plotting MACD and RSI
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot MACD
    ax1.plot(close.index, close['macd'], label='MACD', color='blue')
    ax1.plot(close.index, close['macd_signal'], label='Signal Line', color='orange')
    ax1.bar(close.index, close['macd_hist'], label='Histogram', color='green', width=0.7)
    ax1.axhline(0, color='black', linestyle='--',linewidth=0.8)
    ax1.set_title('MACD')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend()

    # Plot RSI
    ax2.plot(close.index, close['rsi'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--')
    ax2.axhline(30, color='green', linestyle='--')
    ax2.set_title('RSI')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('save_image/explore/MACD and RSI plot' )

