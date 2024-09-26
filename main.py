"""
This is the main module responsible for solving the tasks.
To solve each task just run `python main.py`.
"""

import numpy as np
import pandas as pd

from src import acquisition
from src import storage
from src import processing
from src import exploration
from src import forecasting



def acquire_data():
    """
    Acquire the data from the API and save it in the Data File folder.
    Returns
      stock_data:a dict with stock data
      oil_data:a dict with oil data
    """
    stock_data=acquisition.get_stock_price('2019-04-01','2023-04-01')
    oil_data=acquisition.get_oil_price()

    return stock_data,oil_data

def storing(stock_data,oil_data):
    """
     inset the data in the database collection and return the result
    Args:
      database: the db collection to insert
      item: the json/dict data to be inserted
    Returns:
      result: whether the create is ture
      warning: if the data already exist in
      this colleciton, the result will be false 
    """
    client=storage.connect_mongodb()
    db = client['dpa_final']
    stock_collection = db['stock price']
    wti_collection = db['WTI price']

    print('data storage status:')
    storage.create(stock_collection,stock_data)
    storage.create(wti_collection,oil_data)

def proceesing(stock_data,oil_data):
    """
    the whole processing procedure include clean missing, clean outlier,
    clean missing, plot and pre-processing
    Args:
      stock_data: stock data dict from api
      oil_data: WTI data dict from api
    Returns:
      stock_frame: the stock dataframe after whole pre-processing
      oil_frame: the WTI dataframe after whole pre-processing
    """
    print('\n')
    stock_frame,oil_frame=processing.process(stock_data, oil_data)
    return stock_frame,oil_frame

def exploring(stock_frame,oil_frame):
    """
    the whole exploration procedure with plots and indicators
    Args:
      stock_frame: stock data after preprocessing
      oil_frame: WTI data after preprocessing
    Returns:
      plots in folder
    """
    exploration.explore_plot(stock_frame, oil_frame)
    exploration.indicators_calculate_plot(stock_frame)

def forecast(merge_frame,future_frame):
    """
    predict the april stock price with two models and evaluation
    Args:
      stock_frame: stock data after preprocessing
      oil_frame: WTI data after preprocessing
    Returns:
      print evaluations and plots
    """
    print('\n')
    print("start forecast:")
    print('\n')

    predict=forecasting.single_predict(merge_frame,future_frame,150)
    forecasting.evaluate_model(predict,future_frame,'single prediction joint plot')

    predict_with_regressor=forecasting.predict_with_regressor(merge_frame,future_frame,150)
    forecasting.evaluate_model(predict_with_regressor,future_frame,'prediction with regressors joint plot')


def agent_decision_making(merge_frame,future_frame):
    """
    test the decision agent with april predict data
    Args:
      stock_frame: stock data after preprocessing
      oil_frame: WTI data after preprocessing
    Returns:
      output the advise history and the overall profit
    """
    df=merge_frame[['stock_price']].reset_index()
    budget=10000
    stock=0
    trade_frame=pd.DataFrame(columns=['date','trade'])


    print('test the trade agent in april: ')
    for _ in range(30):
        date,trade,budget,stock=forecasting.agent_advisor(df,budget,stock,future_frame,150)
        new_row = {'date': date, 'trade': trade}
        trade_frame.loc[len(trade_frame)] = new_row

    future=future_frame.reset_index()
    future.rename(columns={'ds':'date'},inplace=True)
    trade=pd.merge(future,trade_frame,on='date',how='left')

    final_stock=np.sum(trade['trade'])

    budget=100000
    cost=0
    last_price=trade.iloc[-1]['actual']

    trade['cost']=trade['actual']*trade['trade']#calculate the money used and get

    cost=np.sum(trade['cost'])

    profit=-cost+final_stock*last_price

    print("final profit:", profit)

    print(trade) 

def main():
    """
    Main function to run the whole project
    """
    # acquire the necessary data
    stock_data,oil_data=acquire_data()

    # store the data in MongoDB
    storing(stock_data,oil_data)

    # format, project and clean the data
    stock_frame,oil_frame=proceesing(stock_data,oil_data)

    # show your findings
    exploring(stock_frame,oil_frame)
    #change the dataframe for further using
    df=stock_frame[['date','close','log_volume']]
    oil=oil_frame[['date','value']]
    merge_frame = pd.merge(df,oil,on='date',how='left')
    merge_frame.rename(columns={'close': 'stock_price'}, inplace=True)
    merge_frame.rename(columns={'value': 'WTI'}, inplace=True)
    merge_frame.set_index('date',inplace=True)

    stock_data=acquisition.get_stock_price('2023-04-01','2023-05-01')

    test_frame=pd.DataFrame(stock_data)
    test_frame.rename(columns={'datetime':'ds'},inplace=True)
    test_frame.rename(columns={'close':'actual'},inplace=True)
    test_frame['ds']=pd.to_datetime(test_frame['ds'])
    test_frame.set_index('ds',inplace=True)
    future_frame=test_frame[['actual']]
    # create a model and train it, visualise the results
    forecast(merge_frame,future_frame)

    #test the agent for april and print the advise reult with profit
    agent_decision_making(merge_frame,future_frame)



if __name__ == "__main__":
    main()
