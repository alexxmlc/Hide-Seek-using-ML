import numpy as np
import random
import os
from keras.models import load_model
from brain import create_brain
from collections import deque

class Agent:
    def __init__(self, start_x, start_y, color, role):
        self.x = start_x
        self.y = start_y
        self.start_x = start_x
        self.start_y = start_y
        self.color = color
        self.active_brain = create_brain()
        self.target_brain = create_brain()
        self.memory = []
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.loss_history = []
        self.role = role
        self.frames = deque(maxlen=4)
        self.visited_tiles = set()
        
    def remember(self, new_mem):
        if len(self.memory) == 10000:
            self.memory.pop(0)
        self.memory.append(new_mem)
        
    def move(self, dx, dy, env, target=None):
        new_x = self.x + dx
        new_y = self.y + dy
        reward = 0
        
        if self.role == "seeker":
            # hits it's head in the wall
            if env.grid[new_y, new_x] == 1 or env.grid[new_y, new_x] == 2:
                reward -= 0.75
            
            # caught the hider
            elif target  is not None and new_x == target.x and new_y == target.y:
                reward += 1.0
                self.x = new_x
                self.y = new_y
                return reward, True
                
            
            # wastes time on empty cells
            elif env.grid[new_y, new_x] == 0:
                reward -= 0.05
                self.x = new_x
                self.y = new_y
                
        elif self.role == "hider":
             # hits it's head in the wall
            if env.grid[new_y, new_x] == 1 or env.grid[new_y, new_x] == 2:
                reward -= 0.75
            
            # gets caught
            elif target  is not None and new_x == target.x and new_y == target.y:
                reward -= 1.0
                self.x = new_x
                self.y = new_y
                return reward, True
                
            
            # wastes time on empty cells
            elif env.grid[new_y, new_x] == 0:
                reward += 0.05
                self.x = new_x
                self.y = new_y
                
        return reward, False
            
    def reset(self, env, target):
        self.x = self.start_x
        self.y = self.start_y
        view = self.observe(env, target=target)
        self.visited_tiles.clear()
        for i in range(4):
            self.frames.append(view)
        
    def observe(self, env, vision_radius=2, target=None):
        vision_size = vision_radius * 2 + 1
        view = np.ones((vision_size, vision_size), dtype=int)     
           
        # 3x3 / 4x4 / 5x5
        min_y = max(0, self.y - vision_radius)
        max_y = min(env.grid.shape[0], self.y + vision_radius + 1)
        min_x = max(0, self.x - vision_radius)
        max_x = min(env.grid.shape[1], self.x + vision_radius + 1)
        
        # put that 3x3 / 4x4 / 5x5 on the 5x5
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
    
    def think_and_move(self, env, target):
        view = self.get_state()
        reshaped_view = np.reshape(view, (1, 4, 5, 5, 1))
        predictions = self.active_brain(reshaped_view, training=False).numpy()
        random_number = random.random()
        
        # 0 -> up / 1 -> down / 2 -> left / 3 -> right
        if random_number <= self.epsilon:
            best_decision = random.randint(0, 3)
        else:
            best_decision = np.argmax(predictions[0]) 
        
        if best_decision == 0:
            dx, dy = 0, -1
        elif best_decision == 1:
            dx, dy = 0, 1
        elif best_decision == 2:
            dx, dy = -1, 0
        elif best_decision == 3:
            dx, dy = 1, 0
            
        reward, done = self.move(dx, dy, env, target)
        
        if reward == -0.05:
            current_pos = (self.x, self.y)
            if current_pos in self.visited_tiles:
                reward -= 0.05  # Wiggle Penalty: punish for its own footprints
            else:
                reward += 0.02  # Explorer Reward: Give a tiny reward hit for finding a new tile
                self.visited_tiles.add(current_pos)
        
        old_state = reshaped_view
        fresh_view = self.observe(env, target=target)
        self.frames.append(fresh_view)
        new_state = np.reshape(self.get_state(), (1, 4, 5, 5, 1))
        
        new_mem = (old_state, best_decision, reward, new_state, done)
        
        self.remember(new_mem)
        return reward
        
        
    def train(self, batch_size):
        random_memories = random.sample(self.memory, batch_size)
        gamma = 0.95
        
       
        old_states = []
        new_states = []
        
        # extract the matrices 
        for old_state, best_decision, reward, new_state, done in random_memories:
            old_states.append(old_state[0]) 
            new_states.append(new_state[0])
            
        old_states = np.array(old_states) # (64, 5, 5, )
        new_states = np.array(new_states)
        
        # the prediction optimization (2 predict calls instead of 128)
        current_scores = self.active_brain.predict(old_states, verbose=0)
        future_scores = self.target_brain.predict(new_states, verbose=0)
        
        # apply Bellman ecuation
        X = old_states
        Y = np.copy(current_scores)
        
        for i, (old_state, best_decision, reward, new_state, done) in enumerate(random_memories):
            if done is True:
                target = reward
            else:
                target = reward + gamma * np.max(future_scores[i])
                
            Y[i][best_decision] = target
            
        # final train
        history = self.active_brain.fit(X, Y, batch_size=batch_size, verbose=0)
        self.loss_history.append(history.history['loss'][0])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        self.target_brain.set_weights(self.active_brain.get_weights())
    
    def save_model(self, file_path):
        self.active_brain.save(file_path)
        
    def load_model(self, file_path):
        if os.path.exists(file_path):
            self.active_brain = load_model(file_path) 
            self.epsilon = 0.01
            print("Brain loaded")
        else:
            print("File not yet created")
            
    def get_state(self):
        # Stack the 4 frames chronologically
        stacked = np.stack(self.frames, axis=0)
        # Add the channel dimension for the Conv2D: shape becomes (4, 5, 5, 1)
        return np.expand_dims(stacked, axis=-1)
        
            
            

    # def train(self, batch_size):
    #     random_memories = random.sample(self.memory, batch_size)
    #     gamma = 0.95
    #     X = []
    #     Y = []
    #     start_time = time.time()
        
    #     for old_state, best_decision, reward, new_state, done in random_memories:
    #         current_scores = self.brain.predict(old_state, verbose=0)
    #         future_scores = self.brain.predict(new_state, verbose=0)
            
    #         if done is True:
    #             target = reward
    #         else:
    #             target = reward + gamma * np.max(future_scores)
                
    #         current_scores[0][best_decision] = target
    #         X.append(old_state[0])
    #         Y.append(current_scores[0])
            
    #     mid_time = time.time()
            
    #     self.brain.train_on_batch(np.array(X), np.array(Y)) 
    #     self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    #     final_time = time.time()
        
    #     print(f"Bucla FOR (predictii): {mid_time - start_time:.4f} secunde")
    #     print(f"Train on Batch: {final_time - mid_time:.4f} secunde")
    #     print("-" * 30)
            
        