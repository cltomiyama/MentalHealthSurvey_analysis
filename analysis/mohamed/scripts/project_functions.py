import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
import matplotlib.pyplot as plt

def load_and_process(path="C:/Users/alraw/Documents/COSC301/survey.csv"):
    
    df1 = (
          pd.read_csv("C:/Users/alraw/Documents/COSC301/survey.csv")
          .drop(columns=['state','self_employed','work_interfere','coworkers','supervisor','mental_vs_physical','anonymity','obs_consequence','mental_health_interview','care_options','phys_health_interview','comments'])
          .dropna(subset=['family_history','no_employees', 'remote_work','mental_health_consequence','phys_health_consequence'])
          .reset_index(drop=False)
          .drop(['index'],axis=1)
    )
    

    return df1