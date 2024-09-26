""" the module is for the forecast and agent part of the daps-final

the module use single model and model with 
regressors to predict the future stock price,
the agent could make decision on stock trading

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def predict_nextday(train,n:int):
    """
    predict the value with prophet for next day
    Argss:
      train: training data of prophet :{ds:,y:}
      n: number of history data used for tarining
    Return:
      forecast: the forecast history of prophet
    """
    model = Prophet()
    model.fit(train.tail(n))  #fit the last n part of the training data
    future=model.make_future_dataframe(periods=1)
    forecast = model.predict(future)

    last_row = forecast.iloc[-1]
    last_date = last_row['ds']
    last_forecast_value = last_row['yhat']
    new_row = {'ds': last_date, 'y': last_forecast_value}
    #add the new predict data at the end of training for next prediction
    train.loc[len(train)] = new_row 

    return forecast

def plot_predictions(forecast,train,test,n,save_name):
    """
    Function to plot the predictions
    Args:
      forecast: the forecast history of prophet
      train: the training data of model
      test: the actual stock price for testing
      save_name: the name of the plot file
    Return:
      f,ax: the plot objects
      
    """

    # create a figure and axis for the plot with a specific figsize
    f, ax = plt.subplots(figsize=(14, 8))

    # extract the data from the forecast corresponding to the training period (up to '2020-07-31')

    train.index = pd.to_datetime(train.ds)
    train=train.loc[:'2023-03-31',:]
    train=train.tail(n-27)

    # plot actual values as black markers
    ax.plot(train['ds'], train.y, 'ko', markersize=3)

    forecast.index = pd.to_datetime(forecast.ds)

    forecast1=forecast.loc[:'2023-03-31',:]

    # plot predicted values as a blue line
    ax.plot(forecast1.index, forecast1.yhat, color='steelblue', lw=0.5)

    # fill the uncertainty interval with a light blue color
    ax.fill_between(forecast1.index, forecast1.yhat_lower, forecast1.yhat_upper, color='steelblue', alpha=0.3)

    # extract the data from the forecast corresponding to the testing period (from '2020-08-01' onwards)
    forecast2=forecast.loc['2023-04-01':'2023-04-30',:]

    # plot actual values as red markers
    ax.plot(test.index, test.actual, 'ro', markersize=3)

    # plot the predicted values as a coral line
    ax.plot(forecast2.index, forecast2.yhat, color='coral', lw=0.5)

    # fill the uncertainty interval with a light coral color
    ax.fill_between(forecast2.index, forecast2.yhat_lower, forecast2.yhat_upper, color='coral', alpha=0.3)

    # add a vertical dashed line to mark the separation point between training and testing data
    ax.axvline(forecast.loc['2023-04-01', 'ds'], color='k', ls='--', alpha=0.7)

    # add gridlines
    ax.grid(ls=':', lw=0.5)

    ax.set_title(save_name)

    save_path = os.path.join('save_image/forecast/', save_name)
    f.savefig(save_path)

    return f, ax

def single_predict(merge_frame,future_frame,n):
    """
    Function to predict the stock price for april
    Args:
      merge_frame: the merged dataframe for training
      {date, stock_price,log_volume,WTI}
      future_frame: the actual price dataframe
      {date,actual}
      n: number of history data used for tarining 
    Return:
      single_result: the predicition in april
    """
    # rename th edataframe to fit prophet
    prophet_data = merge_frame.loc['2021-05-01':].copy()
    prophet_stock = prophet_data[['stock_price']].reset_index()
    prophet_stock.rename(columns={'date':'ds'},inplace=True)
    prophet_stock.rename(columns={'stock_price':'y'},inplace=True)
    for _ in range(30):
        forecast=predict_nextday(prophet_stock,n)

    plot_predictions(forecast,prophet_stock,future_frame,n,'single model forcast plot')
    single_result=prophet_stock.copy()
    single_result.set_index('ds',inplace=True)
    single_result.rename(columns={'y':'predict'},inplace=True)
    single_result=single_result.loc['2023-04-01':'2023-04-30']

    return single_result
  
def evaluate_model(predict, actual,save_name):
    """
    Function to evaluate the model 
    Args:
      predict: the predict result 
      {ds, predict}
      actual: the actual price dataframe
      {ds,actual}
      save_name: the file name to save the joint plot 
    
  """
    merged = pd.merge(predict, actual, on='ds',how='right')
    mse = mean_squared_error(merged['actual'], merged['predict'])
    mae = mean_absolute_error(merged['actual'], merged['predict'])
    r2 = r2_score(merged['actual'], merged['predict'])
    merged['residual']=merged['actual']-merged['predict'] #add residual column

    mean_residual = np.mean(merged['residual'])
    median_residual = np.median(merged['residual'])
    skewness_residual = pd.Series(merged['residual']).skew()
  
    print('\n')
    print('result of '+save_name+' :')
    print(f"Mean squred error: {mse}")
    print(f"mean absolute error: {mae}")
    print(f"R-squared: {r2}")

    print("Mean of residuals:", mean_residual)
    print("Median of residuals:", median_residual)
    print("Skewness of residuals:", skewness_residual)
    print('\n')

    g=sns.jointplot(x='actual', y='predict', data=merged, kind='reg', color="b")
    # set the width and height of the figure
    g.fig.set_figwidth(8)
    g.fig.set_figheight(8)
    # access the first subplot in the figure (histograms) and display the correlation coefficient
    ax = g.fig.axes[0]
    ax.grid(ls=':')

    # add gridlines
    plt.suptitle(save_name)
    save_path = os.path.join('save_image/forecast/', save_name)
    plt.savefig(save_path)

def predict_nextday_with_regressor(train,log_volume,WTI,n:int):
    """
    predict the value with prophet for next day with regressors
    Argss:
      train: training data of prophet :{ds:,y:}
      log_volume: the regressor data{ds:,y:}
      WTI:the wti data{ds:,y:}
      n: number of history data used for tarining
    Return:
      forecast: the forecast history of prophet
    """
    model = Prophet()
    model.add_regressor('log_volume')
    model.add_regressor('WTI')
    model.fit(train.tail(n))
    future=model.make_future_dataframe(periods=1)
    #first predict the two regressors for next day
    predict_nextday(log_volume,n)
    predict_nextday(WTI,n)

    #add the predict regressors values to future dataframe
    future=pd.merge(future, log_volume,on='ds', how='left')
    future.rename(columns={'y':'log_volume'},inplace=True)
    future=pd.merge(future, WTI,on='ds', how='left')
    future.rename(columns={'y':'WTI'},inplace=True)
    predict=future.iloc[-1]
    volume_value=predict['log_volume']
    wti_vlaue=predict['WTI']

    forecast = model.predict(future)

    last_row = forecast.iloc[-1]
    last_date = last_row['ds']
    last_forecast_value = last_row['yhat']

    #add the row to train for next training
    new_row =  {'ds': last_date, 'y': last_forecast_value,'log_volume':volume_value,'WTI':wti_vlaue}
    train.loc[len(train)] = new_row

    return forecast

def predict_with_regressor(merge_frame,future_frame,n):
    """
    Function to predict the stock price for april
    Args:
      merge_frame: the merged dataframe for training
      {date, stock_price,log_volume,WTI}
      future_frame: the actual price dataframe
      {date,actual}
      n: number of history data used for tarining 
    Return:
      single_result: the predicition in april
      
    """
    prophet_data = merge_frame.loc['2021-05-01':].copy()

    #convert three data into prophet format {ds:,y:}
    prophet_data.reset_index()
    prophet_stock=prophet_data[['stock_price','log_volume','WTI']].reset_index()
    prophet_stock.rename(columns={'date':'ds'},inplace=True)
    prophet_stock.rename(columns={'stock_price':'y'},inplace=True)

    prophet_vloume = prophet_data[['log_volume']].reset_index()
    prophet_vloume.rename(columns={'date':'ds'},inplace=True)
    prophet_vloume.rename(columns={'log_volume':'y'},inplace=True)

    prophet_wti = prophet_data[['WTI']].reset_index()
    prophet_wti.rename(columns={'date':'ds'},inplace=True)
    prophet_wti.rename(columns={'WTI':'y'},inplace=True)
    for _ in range(30):
        forecast=predict_nextday_with_regressor(prophet_stock,prophet_vloume,prophet_wti,n)


    plot_predictions(forecast,prophet_stock,future_frame,n,'model with regressors forcast plot')
    single_result=prophet_stock.copy()
    single_result=single_result[['ds','y']]
    single_result.set_index('ds',inplace=True)
    single_result.rename(columns={'y':'predict'},inplace=True)
    single_result=single_result.loc['2023-04-01':'2023-04-30']

    return single_result

def agent_advisor(df,current_budget,stock_count,future_frame,n):
    """
    Function to advise on stock trade
    Args:
      df: dataframe of history stock price
      current_budget: budget
      stock_count: commonly 0
      future_frame: to check the trade day
      n: use n history data for training
    Return:
      last_date: the date of the advise
      trade_amount: the advise trade amout,
        '+' for buy,'-' for sell
      current_budget:the budget after trade
      stock_count: the stock count after trading
      
    """
    # Function to calculate Exponential Moving Average 
    def calculate_ema(prices, days):
        return prices.ewm(span=days, adjust=False).mean()
  
    prophet_data = df.copy()
    prophet_stock = prophet_data[['date','stock_price']].reset_index()
    prophet_stock.rename(columns={'date':'ds'},inplace=True)
    prophet_stock.rename(columns={'stock_price':'y'},inplace=True)
  
    #predict the next day stock price
    predict_nextday(prophet_stock,n)
    last_row = prophet_stock.iloc[-1]
    last_date = last_row['ds']
    last_forecast_value = last_row['y']
    new_row = {'date': last_date, 'stock_price': last_forecast_value}
    df.loc[len(df)] = new_row

    # Calculate MACD for new day
    short_ema = calculate_ema(df['stock_price'], 12)
    long_ema = calculate_ema(df['stock_price'], 26)
    df['macd'] = short_ema - long_ema
    df['macd_signal'] = calculate_ema(df['macd'], 9)
    df['macd_hist'] = df['macd'] - df['macd_signal'].copy()
  
    # calculate whether to buy or cell, 1 for buy, 0 for cell
    df['TradeSignal'] = np.where(df['macd'] > df['macd_signal'], 1, 0)

    #only advise on trading day
    print(last_date)
    if df['TradeSignal'].iloc[-1] == 1 and last_date in future_frame.index :
        trade_amount=int(0.1*current_budget/last_forecast_value)
        current_budget=current_budget-trade_amount*last_forecast_value
        stock_count=stock_count+trade_amount
        print("trade: ",trade_amount)


    elif df['TradeSignal'].iloc[-1] == 0 and last_date in future_frame.index:
        trade_amount=-1*int(0.3*stock_count)
        stock_count=stock_count+trade_amount
        current_budget=current_budget-trade_amount*last_forecast_value
        print("trade: ",trade_amount)


    # not in the trading date, the result is 0
    elif last_date not in future_frame.index:
        trade_amount=0
        print('trade: 0')

    return last_date,trade_amount,current_budget,stock_count
  