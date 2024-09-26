""" the module is for the processing part of the daps-final

the module clean the data with missing values and outliers,
plot the data and pre-process the data

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def fill_missing(df: pd.DataFrame, column_name:str):
    """
    fill the missing value with pd.interpolate methid
    Argss:
      df: the dataframe with missing value
      column_name: the column of the df with missing values
    Return:
      df: dataframe cleaned
    """
    df[column_name]=df[column_name].interpolate(method='cubic') 
    return df

def iqr_outlier_replace(df, column):
    """
    detect the outlier with iqr method
    Argss:
      df: the dataframe with moutlier
      column_name: the column of the df with outliers
    Return:
      df: dataframe replaced the outlier with nan
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    outlier_condition = ((df[column] < (q1 - 1.5 * iqr)) | (df[column] > (q3 + 1.5 * iqr)))

    df.loc[outlier_condition, column] = np.nan #replace the outlier with nan

    return df

def boxplot_save(df, column, filename):
    """
    Returns the boxplot of the data from specific column of a dataframe
    Args:
     df: the dataframe that contains the data to be processed.
     column: the label of the column which contains the value that needs to be processed.
     filename: the name of the boxplot file.
    """
    plt.clf()
    sns.boxplot(df[column], orient="v")
    plt.title(f"{filename}")
    save_path = os.path.join('save_image/visualisation', filename)
    plt.savefig(save_path)

def plot_data(df, column, filename):
    """
    Returns the plt of the data against the data from specific column of a dataframe
    Args:
     df: the dataframe that contains the data to be processed.
     column: the label of the column which contains the value that needs to be processed.
     filename: the name of the plot file.
    """
    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(df['date'],df[column])
    ax.grid(ls="--", c="k", alpha=0.2)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{filename}")
    ax.set_title(f"{filename}")
    save_path = os.path.join('save_image/visualisation', filename)
    fig.savefig(save_path)


def process(stock_data, oil_data):
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
    stock_frame=pd.DataFrame(stock_data)
    stock_frame.rename(columns={'datetime': 'date'}, inplace=True)
    stock_frame['date']=pd.to_datetime(stock_frame['date'])
    oil_frame=pd.DataFrame(oil_data)
    oil_frame['date']=pd.to_datetime(oil_frame['date'])

    plot_data(oil_frame,'value','WTI price with missing value')

    #clean the missing values in two dict
    for column in stock_frame.loc[:, stock_frame.columns != 'date']:
        stock_frame=fill_missing(stock_frame,column)
    oil_frame=fill_missing(oil_frame,'value')

    boxplot_save(stock_frame,'volume','volume boxplot before iqr outlier cleaning')
    boxplot_save(oil_frame,'value','WTI boxplot before iqr outlier cleaning') 
    #clean the outliers
    for column in stock_frame.loc[:, stock_frame.columns != 'date']:
        stock_frame=iqr_outlier_replace(stock_frame,column)
    oil_frame=iqr_outlier_replace(oil_frame,'value')  
    #fill the missing value of outliers
    stock_frame=fill_missing(stock_frame,'volume')
    oil_frame=fill_missing(oil_frame,'value')

    boxplot_save(stock_frame,'volume','volume boxplot after iqr outlier cleaning')
    boxplot_save(oil_frame,'value','WTI boxplot after iqr outlier cleaning')

    plot_data(stock_frame,'close','stock price over time')
    plot_data(stock_frame,'volume','volume over time')
    plot_data(oil_frame,'value','WTI price over time')

    plt.clf()
    stock_frame.boxplot()
    plt.title('stock data boxplot')
    plt.savefig('save_image/visualisation/boxplot together.png')
    #plot the data distributions
    plt.clf()
    for i, col in enumerate(stock_frame.loc[:, stock_frame.columns != 'date']): 
        plt.subplot(2, 3, i+1)
        plt.hist(stock_frame[col])
        plt.title(col)

    plt.subplot(2, 3, 6)
    plt.hist(oil_frame['value'])
    plt.title('WTI')
    plt.tight_layout()
    plt.savefig('save_image/visualisation/histogram of all values')

    stock_frame['log_volume'] = np.log(stock_frame['volume'])
    plot_data(stock_frame,'log_volume','log_volume')

    return stock_frame, oil_frame
