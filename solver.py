# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 20:56:57 2019

@author: Wen Jie
"""

import pandas as pd
#from sklearn.model_selection import train_test_split
#from keras.models import Sequential
#from keras.layers import Dense, Activation,Dropout
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from IPython.display import clear_output
import random

df = pd.read_csv('sudoku.csv',nrows = 50000)

#X_train, X_test, y_train, y_test = train_test_split(df.quizzes, df.solutions, test_size=0.3)

#model = Sequential()

#model.add(Dense(81, input_shape=(81,)))
#model.add(Activation('relu'))    
#model.add(Dense(81))
#model.add(Activation('relu'))    
#model.add(Dense(81))
#model.add(Activation('softmax'))

sol = df.iloc[0,1]

grid = np.zeros((9,81,2))

# For finding position of one-hot object
# AI = [1,0]
# Rewards = [0,1]
def findLoc(state, objectindex):
    for i in range(9):
        for j in range(81):
            if (state[i,j,objectindex]==1):
                return i,j
            
def one_hot(sol):
    # Returns the solution in one-hot encoding. For initializing Q table.
    for i,s in enumerate(sol):
    #print(s)

        grid[(int(s)-1),i] = np.array([0,1])
    
    return grid

def init_player(state):
    for i in range(9):
        if state[i,0,1] == 1 :
           state[i,0,0] = 1
    
    return state

def makeMove(state, action):
    
    player_loc = findLoc(state,0)
    
    if player_loc[0] == 8 and action == 1:
        return makeMove(state,np.random.choice([0,2]))
        
    elif (player_loc[0] == 0 and action == 0):
        return makeMove(state,np.random.choice([1,2]))

#    elif player_loc[1] == 0 and action == 2:
#        return makeMove(state,np.random.choice([0,1]))
    
    elif player_loc[1] == 80 and action == 2:
         return makeMove(state,np.random.choice([0,1,2]))

    state[player_loc[0],player_loc[1],0] = 0
    
    actions = [[-1,0],[1,0],[0,1]]
    #e.g. up => (player row - 1, player column + 0)
    print('Before: ',player_loc)
    new_loc = (player_loc[0] + actions[action][0], player_loc[1] + actions[action][1])
    
    print('After: ',new_loc)
    state[new_loc[0],new_loc[1],0] = 1

    new_player_loc = findLoc(state, 0)
   
    return state,new_player_loc

def getReward(state):
    player_loc = findLoc(state, 0)
    
    if (state[player_loc[0],player_loc[1], 1] == 1 ):
        return 10

    else:
        return -1


model = Sequential()
model.add(Dense(1458, init='lecun_uniform', input_shape=(1458,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

model.add(Dense(150, init='lecun_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(3, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

epochs = 1000
gamma = 0.9 #since it may take several moves to goal, making gamma high


epsilon = 1
for i in range(epochs):
    
    sol = df.iloc[i,1]
    state = init_player(one_hot(sol))
    status = 1
    #while game still in progress
    while(status == 1):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1,1458), batch_size=9)

        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,3)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
        #Take action, observe new state S'
        
        new_state,_ = makeMove(state, action)
        #Observe reward
        reward = getReward(new_state)
        #Get max_Q(S',a)
        newQ = model.predict(state.reshape(1,1458), batch_size=9)
        maxQ = np.max(newQ)
        y = np.zeros((1,3))
        y[:] = qval[:]
        if reward == -1: #non-terminal state
            update = (reward + (gamma * maxQ))
        else: #terminal state
            update = reward
        y[0][action] = update #target output
        print("Game #: %s" % (i,))
        model.fit(state.reshape(1,1458), y, batch_size=1, nb_epoch=1, verbose=1)
        state = new_state
        a,b = findLoc(state,0)
        if (reward != -1 and b == 80):
            status = 0
        clear_output(wait=True)
    if epsilon > 0.1:
        epsilon -= (1/epochs)