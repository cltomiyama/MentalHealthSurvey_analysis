# import commands
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

# functions

def load_and_process(url_or_filepath):
    
    # method chain 1: dataset loading and first round of filtering rows and columns
    
    df1 = (
    pd.read_csv(url_or_filepath)
       .loc[:, ['self_employed', 'family_history', 'treatment', 'work_interfere', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help']]
       .loc[lambda row : row['self_employed'].str.find('Yes')== -1]
       .loc[lambda row : row['tech_company'].str.find('No') == -1]
      )
    
    # method chain 2: handling missing data and removing more undesired columns
    
    df2 = (
    df1.reset_index()
        .drop(['index', 'self_employed', 'tech_company'], axis=1)
        .fillna(value='N/A')
      )
    
    # method chain 3: coding response options
    
    df3 = (
    df2.pipe(code_mcq, col='benefits')
    .pipe(code_mcq, col='care_options')
    .pipe(code_mcq, col='wellness_program')
    .pipe(code_mcq, col='seek_help')
    .assign(condition = np.where(df2['work_interfere'] != 'N/A', 'Yes', 'No'))
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

def rel_freq_label(plot,x,y):
    totals = []

    for i in plot.patches:
        totals.append(i.get_height())

    total = sum(totals)

    for i in plot.patches:
        plot.text(i.get_x()+x, i.get_height()+y, \
                str(round((i.get_height()/total)*100, 2))+'%', fontsize=15,
                    color='0.2')
        
def count_rel_freq_df(df,col):
    new_df = (df[col].value_counts().to_frame().rename_axis(col).rename(columns={col:'count'}).reset_index())
              
    new_df = new_df.assign(rel_freq = round(((new_df['count'] / new_df['count'].sum()) * 100),2))

    return new_df

def make_relfreq_col(df):
    df['rel_freq'] = round(((df['count'] / df['count'].sum()) * 100),2)
    
    return df

def rel_freq_within_grp(df):
    df['withingrp_relfreq'] = df.groupby(level=0).apply(lambda x: 100*x/x.sum())
    df['withingrp_relfreq'] = df['withingrp_relfreq'].apply(lambda x : round(x,2))

    return df

class display(object):
    # taken from class notes
    
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)