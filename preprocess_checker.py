#Distributions and Modules
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import distributions
import pandas as pd
import numpy as np
from torch import optim

gamma = 0.9

def Normalization(df):
    df.iloc[:,2:]= df.iloc[:,2:].apply(lambda x: ((x-x.mean()) / (x.std())))
    return df

def preprocessing():
    
    df1 = pd.read_csv("CMAPSSData/train_FD001.txt", header=None, delimiter=' ')
    df2 = pd.read_csv("CMAPSSData/train_FD002.txt", header=None, delimiter=' ')
    df3 = pd.read_csv("CMAPSSData/train_FD003.txt", header=None, delimiter=' ')
    df4 = pd.read_csv("CMAPSSData/train_FD004.txt", header=None, delimiter=' ')
    
    df = pd.concat([df1, df2])
    #Normalize the data
    df = Normalization(df)
    
    #Drop the columns which has all values as Nan
    df.dropna(axis=1, how='all', inplace=True)
    
    #Get Rewards for each time step : 0 except last time step where reward is -100
    df['Counter'] = df.index
    lastRowIndex = df.groupby(0).last().Counter.tolist()
    df['reward'] = df['Counter'].apply(lambda x : -100 if x in lastRowIndex else 0 )
    df.drop(columns=['Counter'],inplace=True)
    
    #Rename columns
    df.rename(columns={0: "machine", 1: "time"}, inplace=True)
    
    #Calculate Monte Carlo Value for each row
    df1 = df.groupby('machine').last()[['time']].reset_index()
    df = pd.merge(df, df1, on = 'machine', how = 'left').rename(columns ={'time_x':'time','time_y':'lastTimeStamp'})
    df['MC_Val'] = (gamma ** (df['lastTimeStamp'] - df['time'] )) * (-100)
    df = df.drop(columns='lastTimeStamp')
    
    return df




df = preprocessing()
df.to_csv("cmapss_datasets/training.csv")







