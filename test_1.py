import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def find_outlinder(data):
    a = str(input('Nhap cot ban mua tim outlinder:'))
    df_outlinder = data[a]
    Q1 = df_outlinder.quantile(0.25)
    Q3 = df_outlinder.quantile(0.75)
    IQR = Q3 - Q1
    outlinder = df_outlinder[(df_outlinder > IQR*1.5 + Q3) | (df_outlinder < Q1 - 1.5*IQR)]
    return outlinder

data_frame = pd.read_csv(os.path.join('machine_learning','data.csv'))
print(data_frame)  
data_frame.info()
drop_set = ['Duration','Date','Calories']
pulse_maxpulse = data_frame.drop(drop_set,axis= 1)
fig,ax = plt.subplots(2,2)
sns.histplot(data= pulse_maxpulse,ax = ax[0,1])
sns.boxplot(data= pulse_maxpulse,ax= ax[1,1])
sns.heatmap(data= pulse_maxpulse,ax = ax[1,0])
print(pulse_maxpulse)
plt.show()
outlinde =  find_outlinder(data= pulse_maxpulse)
outlinder_index = outlinde.index
print(outlinder_index)
row_outlinder = pulse_maxpulse.iloc[outlinder_index]
print(row_outlinder)
print(sns.get_dataset_names)