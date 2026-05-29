import numpy as np
from settings import GRID_WIDTH, GRID_HEIGHT

class Environment:
    def __init__(self):
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.build_map()

    def build_map(self):
        # ARENA BORDERS 
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
        # TOP-LEFT BARRICADE (L-Shape)
        # y from 4 to 8, x is 4
        self.grid[4:9, 4] = 1   
        # y is 8, x from 4 to 9
        self.grid[8, 4:10] = 1  
        
        # BOTTOM-LEFT PILLARS (Cover for the Hider)
        self.grid[14:17, 5:7] = 1
        self.grid[14:17, 9:11] = 1
        
        # THE CENTRAL BUNKER
        # Left and Right walls of the bunker
        self.grid[8:15, 14] = 1 
        self.grid[8:15, 20] = 1 
        # Top wall of the bunker
        self.grid[8, 14:21] = 1 
        # Bottom wall of the bunker (with a gap for doors)
        self.grid[14, 14:16] = 1 
        self.grid[14, 19:21] = 1 
        
        # BUNKER DOORS 
        # Fills the gap in the bottom bunker wall
        self.grid[14, 16:19] = 2 
        
        # TOP-RIGHT DIVIDER
        self.grid[2:6, 17] = 1   

    def toggle_door(self, x, y):
        if self.grid[y, x] == 2:
            self.grid[y, x] = 0  
        elif self.grid[y, x] == 0:
            self.grid[y, x] = 2