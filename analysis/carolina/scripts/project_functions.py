import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling
from pandas_profiling import ProfileReport

def load_and_process_ct(csv_file):
    df_raw= (
        pd.read_csv(csv_file)
        .rename(columns= {'no_employees':'employees', 'mental_health_consequence':'consequence'}))
    global df
    df= df_raw[["employees", "consequence", "coworkers", "supervisor"]]
    
    global noe, mhc, co, sup
    noe = df["employees"]
    mhc = df["consequence"]
    co = df["coworkers"]
    sup = df["supervisor"]

    global noe_q, mhc_q, co_q, sup_q
    noe_q = "How many employees does your company or organization have?"
    mhc_q = "Do you think that discussing a mental health issue with your employer would have negative consequence?"
    co_q = "Would you be willing to discuss a mental health issue with your coworker?"
    sup_q = "Would you be willing to discuss a mental health issue with your direct supervisor?"
    
    return df

def count_table_ct(variable_name):
    variable = eval(variable_name)
    
    if variable_name == "noe":
        q = noe_q
        ind = ['1', '2', '5', '3', '0', '4']
    elif variable_name == "mhc":
        q = mhc_q
        ind = ['1', '2', '0']
    elif variable_name == "co":
        q = co_q
        ind = ['2', '1', '0']
    else:
        q = sup_q
        ind = ['0', '1', '2']
    
    print("Question:", q)
    
    variable_count1= variable.value_counts().rename_axis('Answer').reset_index(name= 'Counts')
    variable_count2= variable_count1.set_index(pd.Index(ind)).sort_index()
    total= variable_count2.apply(np.sum)
    total['Answer']= 'Total'
    variable_count= variable_count2.append(pd.DataFrame(total.values, index= total.keys()).T, ignore_index= True)
    return variable_count

def count_plot_ct(variable_name):
    variable= eval(variable_name)
    
    if variable_name == "noe":
        plt.figure(figsize= (10,6))
        title= noe_q
        order= ("1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000")
        plt.ylim(0,400)
    
    elif variable_name == "mhc":
        plt.figure(figsize= (8,5))
        title= mhc_q
        order= ("Yes", "No", "Maybe")
        plt.ylim(0,600)
   
    elif variable_name == "co":
        plt.figure(figsize= (8,5))
        title= co_q
        order= ("Yes", "No", "Some of them")
        plt.ylim(0,900)
    
    else:
        plt.figure(figsize= (8,5))
        title= sup_q
        order= ("Yes", "No", "Some of them")
        plt.ylim(0,650)
        
    variable_countplot= sns.countplot(x= variable, order= order)
    variable_countplot.set(xlabel= "Answer", ylabel= "Count", title= title)
    for p in variable_countplot.patches:
        variable_countplot.annotate('count = {:}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))
        
def groupby_noe_ct(variable1, variable2):
    
    if variable1 == "consequence":
        ind= [2,0,1]
        indx= [2,1,0,14,13,12,8,7,6,5,4,3,11,10,9,17,16,15]
        variable1_new= 'Mental health consequence:'
        variable2_new= 'Supervisor:'
    elif variable1 == "coworkers":
        ind= [2,0,1]
        indx= [2,0,1,11,9,10,8,6,7,14,12,13,5,3,4,17,15,16]
        variable1_new= 'Coworkers:'
        variable2_new= 'Supervisor:'
    elif variable1 == "supervisor" and variable2 == "consequence":
        ind= [2,1,0]
        indx= [2,0,1,11,9,10,8,6,7,14,12,13,5,3,4,17,15,16]
        variable1_new= "Supervisor:"
        variable2_new= "Mental health consequence:"
    else:
        ind= [2,0,1]
        indx= [2,1,0,14,13,12,8,7,6,5,4,3,11,10,9,17,16,15]
        variable1_new= 'Supervisor:'
        variable2_new= 'Coworkers:'
    
    # Grouping and Cleaning
    df_noe= pd.DataFrame({'':df.groupby(["employees", variable1, variable2]).size()}).unstack().replace(np.nan, 0)

    # Reordering columns
    cols= df_noe.columns.tolist() 
    cols= [cols[i] for i in ind]
    df_noe= df_noe[cols]

    # Reordering index
    df_noe= df_noe.iloc[indx]

    # Formatting
    df_noe= df_noe.rename_axis(index={'employees':'Number of employees:', variable1:variable1_new}, 
                               columns={variable2:variable2_new})
    df_noe= (df_noe.style.set_properties(subset= df_noe.columns, **{'width':'7em', 'text-align':'center'})
                    .set_table_styles([dict(selector= 'th', props= [('text-align', 'left')])]).format('{0:,.0f}'))
    return df_noe

def chi_sq(table):
    stat, p, dof, expected= chi2_contingency(table)
    print('(p-value =', p, 'df =', dof, ')')
    alpha= 0.05
    if p <= alpha:
        print('Dependent (reject null hypothesis).')
    else:
        print('Independent (fail to reject null hypothesis)')