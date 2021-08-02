import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
import matplotlib.pyplot as plt

def load_and_process(url_or_path_to_csv_file):
    
    df1 = (
          pd.read_csv(url_or_path_to_csv_file)
          .drop(columns=['state','self_employed','work_interfere','coworkers','supervisor','mental_vs_physical','anonymity','obs_consequence','mental_health_interview','care_options','phys_health_interview','comments'])
          .dropna(subset=['family_history','no_employees', 'remote_work','mental_health_consequence','phys_health_consequence'])
          .replace(['Cis Male', 'Mail', 'Make', 'Male', 'Male ', 'Male-ish', 'Malr', 'Male (CIS)', 'Male', 'Man', 'something kinda male?', 'cis male', 'm', 'maile', 'male', 'msle'], 'M')
          .replace(['Femake', 'Female', 'Female ', 'Female (cis)', 'Woman', 'f', 'female'], 'F')
          .replace(['Femake', 'Female', 'Female ', 'Female (cis)', 'Woman', 'f', 'female'], 'F')
          .replace(['Female (trans)', 'Trans-female'], 'MTF')
          .replace(['Androgyne', 'Genderqueer', 'fluid', 'male leaning androgynous', 'queer/she/they'], 'F')
          .reset_index(drop=False)
          .drop(['index'],axis=1)
    )
    
    return df1