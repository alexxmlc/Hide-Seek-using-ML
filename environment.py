import numpy as np
from settings import GRID_WIDTH, GRID_HEIGHT

class Environment:
    def __init__(self):
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.build_map()

    def build_map(self):
        # walls on the edges
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
        # wall with door in the middle
        self.grid[5:10, 10] = 1 
        self.grid[7, 10] = 2    

    def toggle_door(self, x, y):
        if self.grid[y, x] == 2:
            self.grid[y, x] = 0  
        elif self.grid[y, x] == 0:
            self.grid[y, x] = 2