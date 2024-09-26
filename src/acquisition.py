""" a module to acqure the stock data and auxiliary data frame api

the two methods get the stock data and WTI data from two api and chnage the data type for storage


"""
import numpy as np
import requests
from datetime import datetime

stock_key='048768abb3054971bd6b80f5cca61c70'
oil_key='KP5SGMIGI6L5SKJA'
company='AAL'

def get_stock_price(start_date:str, end_date:str):
    """
      get the stock data from twelveData API and change the data type
    Args:
        start_date:start data of the acquire stock data
        end_date: end date of the acquire stock data
        both forms are: YYYY-MM-DD
    Returns:
        stock_data:a dict with stock data
        columns: datetime: str,open: float,
        high: float,low:float,close: float,
        volume: int
    """
    params = {
     'interval':'1day',
     'apikey': {stock_key},
     'symbol':{company},
     'start_date':start_date,
     'end_date':end_date,
     'order':'ASC'
    }
    url='https://api.twelvedata.com/time_series?'

    result = requests.get(url, params, timeout=20)
    response = result.json()
    stock_data=response.pop('values')

    for row in stock_data:
        row['open'] = float(row['open'])
        row['high'] = float(row['high'])
        row['low'] = float(row['low'])
        row['close'] = float(row['close'])
        row['volume'] = int(row['volume'])

    return stock_data

def get_oil_price():
    """
    get the oil data from alphavantage API and change the data type
    Args:
    Returns:
        oil_data:a dict with oil data
        columns: date: value: float
    """
    params = {
     'apikey': {oil_key},
     'function':'WTI',
     'interval':'daily'
    }
    url='https://www.alphavantage.co/query?'
    auxiliary_response = requests.get(url, params, timeout=20)
    abab = auxiliary_response.json()
    #this request returns the history data for 20 years
    # restrict the data to the time period requested
    start_date = datetime(2019, 4, 1)
    end_date = datetime(2023, 3, 31)
    oil_data = [item for item in abab['data'] if start_date \
                <= datetime.strptime(item['date'], '%Y-%m-%d') <= end_date]
    for row in oil_data:
        if row['value'] == '.':
            row['value'] = np.nan  #change '.'to nan for further missing cleaning
        else:
            row['value'] = float(row['value'])

    return oil_data




