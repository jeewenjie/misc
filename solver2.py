
import numpy as np
import pandas as pd


row = 'ABCDEFGHI'
col = '123456789'

some_puzzles = pd.read_csv('sudoku.csv',nrows = 10) # col: puzzle,solution

puzzle = some_puzzles.iloc[0,0] # First row, first col  

grid_names = []
for i in row:
    for j in col:
    	grid_names.append(i+j)

possible_solutions = {} # Define empty dict first.

def possiblities(puzzle,possible_solutions):
    assert len(puzzle) == 81, "Not the correct length."
    for i,s in enumerate(puzzle):
        if s == '0' or s =='.':
            possible_solutions[grid_names[i]] = '123456789'
        else:
	        possible_solutions[grid_names[i]] = s
    return possible_solutions
#	    if isinstance(puzzle,string): 
 #          raise ValueError('Puzzle should be a string.')
     #return possible_solutions
dictionary = possiblities(puzzle,possible_solutions)
print(dictionary)