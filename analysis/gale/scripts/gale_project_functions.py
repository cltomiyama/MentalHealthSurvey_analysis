import pandas as pd
import numpy as np

def load_and_process(url_or_filepath):
    # method chain 1: load data and drop unwanted rows and columns
    
    mh1 = (
        pd.read_csv(url_or_filepath)
           .drop(['Timestamp', 'state', 'family_history', 'coworkers', 'Age', 'comments', 'no_employees', 'Country'], axis='columns')
           .loc[lambda row : row['self_employed'].str.find('Yes')== -1]
           .loc[lambda row : row['tech_company'].str.find('No') == -1]
          )
    
    # method chain 2: recategorizing genders
    
    mh1 = (
        mh1.replace(['Cis Male', 'Mail', 'Make', 'Male', 'Male ', 'Male-ish', 'Malr', 'Male (CIS)', 'Male', 'Man', 'something kinda male?', 'cis male', 'm', 'maile', 'male', 'msle'], 'M')
           .replace(['Femake', 'Female', 'Female ', 'Female (cis)', 'Woman', 'f', 'female'], 'F')
           .replace(['Enby', 'non-binary', 'Agender'], 'NB')
           .replace(['Female (trans)', 'Trans-female'], 'MTF')
           .replace(['Androgyne', 'Genderqueer', 'fluid', 'male leaning androgynous', 'queer/she/they'], 'F')
          )

    # method chain 3: handling missing data
    
    mh1 = (mh1.dropna(axis=0, how='any')
           .dropna(subset=['work_interfere'], axis=0, how='any').reset_index()
           .drop(['index', 'self_employed', 'tech_company'], axis=1)
          )
    
    return mh1