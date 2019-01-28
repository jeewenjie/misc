
import numpy as np
import pandas as pd


row = 'ABCDEFGHI'
col = '123456789'

some_puzzles = pd.read_csv('sudoku.csv',nrows = 10) # col: puzzle,solution

puzzle = some_puzzles.iloc[0,0] # First row, first col  

def possiblities(puzzle):
	assert len(puzzle) == 81, "Not the correct length."
	for s in puzzle:

	    elif isinstance(puzzle,string): 
          raise ValueError('Puzzle should be a string.')
