# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 22:07:06 2019

@author: Wen Jie
"""

import pandas as pd
import numpy as np
from keras.models import load_model

df = pd.read_csv('sudoku.csv',nrows = 10000, skiprows = 50000)
model = load_model('model_final')
model.load_weights('model_final')

# Initialise grid
grid = np.zeros((9,81,2))

# Object index: AI = 0, sudoku number = 1
def findLoc(state, objectindex):
    for i in range(9):
        for j in range(81):
            if (state[i,j,objectindex]==1):
                return i,j
            
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


def one_hot(sol):
    # Returns the solution in one-hot encoding. For initializing Q table.
    for i,s in enumerate(sol):

        grid[(int(s)-1),i] = np.array([0,1])


def init_player(state):
    # Changed this for cases where first column does not have a number
    for i in range(81):
        for j in range(9):
           if state[j,i,1] == 1 :
              state[j,i,0] = 1
    
    return state

def getReward(state):
    player_loc = findLoc(state, 0)
    
    if (state[player_loc[0],player_loc[1], 1] == 1 ):
        return 10

    else:
        return -1
       
def test():
    finalsol = ""
    for i in range(1):
        sol = df.iloc[i]
        state = init_player(one_hot(sol))
        
        old_a,old_b = findLoc(state,0)
        status = 1
        counter = 0
        
        while(status == 1):
           qval = model.predict(state.reshape(1,1458), batch_size=1)
           action = (np.argmax(qval)) #take action with highest Q-value
           state = makeMove(state, action)
           reward = getReward(state)
          
           a,b = findLoc(state,0)
           if b == 80:
              counter += 1
             
           if old_b != b: # First time column change, will record down. 
              finalsol += a 
              old_b = b
            
           if (b == 80 and reward == 10) or (counter == 8):
               state = 0
               finalsol +='.'
    
    return finalsol

finalsol = test()