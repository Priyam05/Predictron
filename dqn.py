import random, numpy, math
import pandas as pd
import copy
import numpy as np

#-------------------- BRAIN ---------------------------
import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()

    def _createModel(self):
        model = Sequential()
        model.add(keras.Input(shape=(self.stateCnt,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.actionCnt, activation='linear'))

        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, verbose=0):
        self.model.fit(x, y, batch_size=x.shape[0], verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCnt)).flatten()

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

C = 25 

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.target_brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)
        self.step_count = 0
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)
        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = self.target_brain.predict(states)
        p_ = self.target_brain.predict(states_)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)

        self.step_count += 1
        if self.step_count == C:
            self.target_brain.model = keras.models.clone_model(self.brain.model)
            self.step_count = 0


#-------------------- ENVIRONMENT ---------------------


# Make list of column names
columns = ["unit number", "time in cycles", "operational setting 1",
                "operational setting 2", "operational setting 3"]
for i in range(1, 22):
    columns.append("sensor measurement %d"% i)   


def make_training_dataframe(txt):
    """
    Input:
        txt: name of training set

    Output: Dataframe
    """
    df = pd.read_csv(txt, sep = " ", header=None)
    df.drop(columns=[26, 27], inplace=True)
    df.columns = columns
    return df


class Environment:
    
    def __init__(self, training_set):

        self.training_set = training_set
        self.df = make_training_dataframe(training_set)
        max_unit = max(set(self.df['unit number']))
        self.currStates = [] 
        self.agents = []
        for idx in range(0, max_unit):
            self.currStates.append(self.df[self.df['unit number'] == idx + 1].index[0])
            self.agents.append(Agent(self.df.shape[1] - 5, 2))

        self.initialStates = copy.copy(self.currStates)

    def reset(self):
        """
        Resets the environment for a new episode.
        """
        self.currStates = copy.copy(self.initialStates)
        self.agents = []
        for idx in range(0, max(set(self.df['unit number']))):
            self.agents.append(Agent(self.df.shape[1] - 5, 2))


    def step(self, a, s, idx):
        """
        Input:
            a: action
            s: current state of agent (index of row in dataframe)
            idx: index of agent in array (unit number)

        Output: next state, reward of taking the action, and boolean indicating if episode is over
        """
        
        # Compute next state; if the action is to continue, advance the row number. Otherwise, restart at initial row
        s_ = s + 1 if a == 0 else self.initialStates[idx]

        # Determine if next state is terminal
        done = False
        if ((idx < len(self.initialStates) - 1) and (s_ == self.initialStates[idx + 1])) or s_ == self.df.shape[0]:
            done = True

        # Compute reward [very random placeholder]
        
        terminal_cycle = 0
        if idx < len(self.initialStates) - 1:
            terminal_cycle = self.initialStates[idx+1] - 1
        else:
            terminal_cycle = self.df.shape[0] - 1

        N = terminal_cycle - self.initialStates[idx] + 1
        health = (N - s + 1) / N
        cost_lost_production = a * health * 0.3
        repair_cost = (a / health) * 0.7
        num_operating = sum([1 for machine in self.currStates if machine != None])

        reward = 1 - cost_lost_production - repair_cost + ((a * num_operating) / (len(self.currStates) - 1))
        
        # return next state, reward, done
        return s_, reward, done


    
    def state_parameters(self, s):
        """
        Input: 
            s: state index

        Output: array of state parameters
        """
        # return sensor measurements for input row
        return np.array(self.df.iloc[s].values)[5:]


    def run(self):
        """
        Runs an episode with all of the environment's agents
        """
        
        self.reset()

        while True: 

            terminated_agents = 0

            for idx in range(0, len(self.agents)):

                #Get the current agent
                agent = self.agents[idx]

                #Get the current state of the agent
                s = self.currStates[idx]   
                if s == None:
                    print("Machine: " + str(s) + " terminated")
                    terminated_agents += 1
                    continue    

                #Get the next action in {0, 1} where 0 represents continuing and 1 represents repairing
                a = agent.act(self.state_parameters(s)) 

                s_, r, done = self.step(a, s, idx)

                if done: # terminal state
                    s_ = None

                agent.observe( (self.state_parameters(s), a, r, self.state_parameters(s_)) )
                agent.replay()            

                self.currStates[idx] = s_

            if terminated_agents == len(self.agents):
                break



#-------------------- MAIN ----------------------------
env = Environment('CMAPSSData/train_FD001.txt')


env.run()


