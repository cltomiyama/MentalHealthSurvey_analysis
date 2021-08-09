# import commands
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

# functions

def load_and_process(url_or_filepath):
    
    # method chain 1: initial filtering of rows and columns
    
    df1 = (
    pd.read_csv(url_or_filepath)
       .loc[:, ['self_employed', 'family_history', 'treatment', 'work_interfere', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help']]
       .loc[lambda row : row['self_employed'].str.find('Yes')== -1]
       .loc[lambda row : row['tech_company'].str.find('No') == -1]
      )
    
    # method chain 2: handling missing data and removing more undesired columns
    
    df2 = (
    df1.dropna(axis=0, how='any')
    .reset_index()
    .drop(['index', 'self_employed', 'tech_company'], axis=1)
      )
    
    # method chain 3: coding response options
    
    df3 = (
    df2.pipe(code_mcq, col='benefits')
    .pipe(code_mcq, col='care_options')
    .pipe(code_mcq, col='wellness_program')
    .pipe(code_mcq, col='seek_help')
    .astype('category')
      )
    
    return df3

def code_mcq(df,col):
    """
    Turns survey response strings into numbers. 'Not sure' or 'Don't know' become 0, 'Yes' becomes 1, and 'No' becomes 2.
    
    Arguments:
    df - chosen dataframe
    col - name of column in which responses are to be converted
    
    Returns:
    Dataframe with columns in which response options have been converted into numbers
    
    """
    for o in df[col].unique():
        if o == 'Not sure' or o == "Don't know":
            df[col] = df[col].replace(o, 0)
            
        if o == 'Yes':
            df[col] = df[col].replace(o, 1)
                
        if o == 'No':
            df[col] = df[col].replace(o, 2)
        
    return df