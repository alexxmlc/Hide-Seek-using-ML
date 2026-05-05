import numpy as np

class Agent:
    def __init__(self, start_x, start_y, color):
        self.x = start_x
        self.y = start_y
        self.start_x = start_x
        self.start_y = start_y
        self.color = color
        
    def move(self, dx, dy, env):
        new_x = self.x + dx
        new_y = self.y + dy
        
        if env.grid[new_y, new_x] == 0:
            self.x = new_x
            self.y = new_y
            
    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        
    def observe(self, env, vision_radius=2, target=None):
        vision_size = vision_radius * 2 + 1
        view = np.ones((vision_size, vision_size), dtype=int)     
           
        # 3x3
        min_y = max(0, self.y - vision_radius)
        max_y = min(env.grid.shape[0], self.y + vision_radius + 1)
        min_x = max(0, self.x - vision_radius)
        max_x = min(env.grid.shape[1], self.x + vision_radius + 1)
        
        # put that 3x3 on the 5x5
        v_min_y = vision_radius - (self.y - min_y)
        v_max_y = vision_radius + (max_y - self.y)
        v_min_x = vision_radius - (self.x - min_x)
        v_max_x = vision_radius + (max_x - self.x)
        
        view[v_min_y:v_max_y, v_min_x:v_max_x] = env.grid[min_y:max_y, min_x:max_x]
        
        if target is not None:
            dx = target.x - self.x
            dy = target.y - self.y
            
            if abs(dx) <= vision_radius and abs(dy) <= vision_radius:
                view[vision_radius + dy, vision_radius + dx] = 9
        
        return view