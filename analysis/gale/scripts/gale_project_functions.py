import pandas as pd
import numpy as np

def load_and_process(url_or_filepath):
    # method chain 1: load data and drop unwanted rows and columns
    
    mh1 = (
        pd.read_csv(url_or_filepath)
           .drop(['Gender', 'Timestamp', 'remote_work', 'phys_health_consequence', 'state', 'phys_health_interview', 'family_history', 'Age', 'comments', 'no_employees', 'Country'], axis='columns')
           .loc[lambda row : row['self_employed'].str.find('Yes') == -1]
           .loc[lambda row : row['tech_company'].str.find('No') == -1]
          )

    # method chain 2: handling missing data
    
    mh1 = (mh1.dropna(axis=0, how='any').reset_index()
           .drop(['index', 'self_employed', 'tech_company'], axis=1)
          )
    
    return mh1