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
    state[player_loc[0],player_loc[1],0] = 0
    
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    #e.g. up => (player row - 1, player column + 0)
    new_loc = (player_loc[0] + actions[action][0], player_loc[1] + actions[action][1])

    state[new_loc[0],new_loc[1],0] = 1

    new_player_loc = findLoc(state, 0)
   
    return state,new_player_loc

grid = one_hot(sol)
grid = init_player(grid)
x,y=findLoc(grid,0)
grid,newloc = makeMove(grid,0)