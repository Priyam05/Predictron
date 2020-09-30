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

class NN_OH(nn.Module):
    def __init__(self, input_size, out_size):
        super(NN_OH,self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,out_size)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NN_reward(nn.Module):
    def __init__(self, input_size):
        super(NN_reward,self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,3)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NN_val(nn.Module):
    def __init__(self, input_size):
        super(NN_val,self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,1)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NN_HH(nn.Module):
    def __init__(self, input_size):
        super(NN_HH,self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,input_size)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Now we have all the required neural networks for the predictron. Lets build the Predictron

class Predictronv3(nn.Module):
    def __init__(self, obs_size, hid_size, k=10):
        super(Predictronv3,self).__init__()
        
        #Instantiate Neural Network for Observation-Hidden State
        self.OH = NN_OH(obs_size, hid_size)
        
        #Instantiate Neural Network for Hidden State - Reward, Value
        self.HR = NN_reward(hid_size)
        
        #Instantiate Neural Network for Hidden State - Val
        self.HV = NN_val(hid_size)
        
        #Instantiate Neural Network for Hidden State - Next Hidden State
        self.HH = NN_HH(hid_size)
        
        #K-step return
        self.k = k
        
    def forward(self, x):
        #Predictron core will output the value estimate for the current observation. We will input x (observation) 
        #and get value estimate. This implementation is for a k-step return which can be extended to TD(lambda) return
        
        #First step: Get the Hidden state for the current observation
        
        x = self.OH(x)
        #Get the reward, lambda, gamma for current hidden state
        reward = self.HR(x)[:,0].reshape(-1,1)
        gamma = self.HR(x)[:,1].reshape(-1,1)
        _lambda = self.HR(x)[:,2].reshape(-1,1)
        
        #Get value of the current hidden state
        val = self.HV(x)
        #print(val.shape)
        #Get glk for 0th step
        glk = (1-_lambda)*val + _lambda* reward
        #print(glk.shape)
        
        #Store gamma and lambda as prev lambda
        prev_gamma = gamma
        prev_lambda = _lambda
        
        #Now run the loop for k steps
        for i in range(1, self.k + 1):
            #Move to next hidden step
            x = self.HH(x)
            
            #Get the reward, lambda, gamma for current hidden state
            reward = self.HR(x)[:,0].reshape(-1,1)
            gamma = self.HR(x)[:,1].reshape(-1,1)
            _lambda = self.HR(x)[:,2].reshape(-1,1)
            
            #Get value of the current hidden state
            val = self.HV(x)
            
            #Calculate the lambda return
            glk += (prev_gamma*prev_lambda) * ((1-_lambda)*val + _lambda* reward)
        

        return glk.reshape(-1,1)

def getXY(data):
    x = torch.tensor(data.iloc[:, 2:-2].values).float()
    y_target = torch.tensor(data.iloc[:,-1].values).float()
    y_target = y_target.reshape(-1,1)
    
    return x, y_target


def preprocessing(path):
    
    df = pd.read_csv(path, header=None, delimiter=' ')
    
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

def train(x, y_target, loss_fn, core, optimizer, n_epochs=50, batch_size=256):

    losses=[]
    for epoch in range(n_epochs):

        # x is our input
        permutation = torch.randperm(x.size()[0])

        for i in range(0,x.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x[indices], y_target[indices]

            # in case you wanted a semi-full example
            outputs = core.forward(batch_x)
            loss = loss_fn(outputs,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    return losses


if __name__ == "__main__":

	
    df1 = preprocessing("CMAPSSData/train_FD001.txt")
    
    df2 = preprocessing("CMAPSSData/train_FD002.txt")
    df3 = preprocessing("CMAPSSData/train_FD003.txt")
    df4 = preprocessing("CMAPSSData/train_FD004.txt")

    x1, y_target1 = getXY(df1)
    x2, y_target2 = getXY(df2)
    x3, y_target3 = getXY(df3)
    x4, y_target4 = getXY(df4)

    #Defining the loss function and Initialising the Predictron core
    k=10

    loss_fn1 = nn.MSELoss()
    loss_fn2 = nn.MSELoss()
    loss_fn3 = nn.MSELoss()
    loss_fn4 = nn.MSELoss()
    
    core1 = Predictronv3(x1.shape[1], 4, k)
    core2 = Predictronv3(x2.shape[1], 4, k)
    core3 = Predictronv3(x3.shape[1], 4, k)
    core4 = Predictronv3(x4.shape[1], 4, k)
    
    optimizer1 = optim.Adam(core1.parameters(), lr = 1e-3)
    optimizer2 = optim.Adam(core2.parameters(), lr = 1e-3) 
    optimizer3 = optim.Adam(core3.parameters(), lr = 1e-3) 
    optimizer4 = optim.Adam(core4.parameters(), lr = 1e-3)  

    losses1 = train(x1, y_target1, loss_fn1, core1, optimizer1)
    losses2 = train(x2, y_target2, loss_fn2, core2, optimizer2)
    losses3 = train(x3, y_target3, loss_fn3, core3, optimizer3)
    losses4 = train(x4, y_target4, loss_fn4, core4, optimizer4)

    print("losses1: ")
    print(losses1)

    print("losses2: ")
    print(losses2)

    print("losses3: ")
    print(losses3)

    print("losses4: ")
    print(losses4)
