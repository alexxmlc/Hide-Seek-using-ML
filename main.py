import pygame
import sys
import random
import matplotlib.pyplot as plt
import datetime
from settings import *
from environment import Environment
from agent import Agent

# initialization
pygame.init()
screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
pygame.display.set_caption("Multi agent Hide & Seek")
clock = pygame.time.Clock()

env = Environment()
seek = Agent(1, 1, RED, role="seeker")
seek.load_model('seeker.keras')

hide = Agent(GRID_WIDTH - 2, GRID_HEIGHT - 2, BLUE, role="hider")

seeker_episode_total_reward = 0.0
seeker_episodes = []
hider_episode_total_reward = 0.0
hider_episodes = []
episode_count = 0
step_count = 0

def randomize_spawns(seek, hide, env):
    # Scan the environment for empty floor tiles (value = 0)
    empty_cells = []
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if env.grid[y, x] == 0:
                empty_cells.append((x, y))
                
    # Pick random starting coordinates
    sx, sy = random.choice(empty_cells)
    hx, hy = random.choice(empty_cells)
    
    # Make sure they don't accidentally spawn on the exact same tile
    while hx == sx and hy == sy:
        hx, hy = random.choice(empty_cells)
        
    # Update the coordinates for both agents before resetting memory
    seek.start_x, seek.start_y = sx, sy
    seek.x, seek.y = sx, sy 
    
    hide.start_x, hide.start_y = hx, hy
    hide.x, hide.y = hx, hy
    
    # Reset the memory
    seek.reset(env, hide)
    hide.reset(env, seek)


randomize_spawns(seek, hide, env)

# main loop
running = True
frame_count = 0

while running:
    
    # seek logic
    seeker_step_reward = seek.think_and_move(env, hide)
    seeker_episode_total_reward += seeker_step_reward
    if len(seek.memory) >= 64 and frame_count % 10 == 0:
        seek.train(64)
            
    # hider logic
    # hider_step_reward = hide.think_and_move(env, seek)
    # hider_episode_total_reward += hider_step_reward
    # if len(hide.memory) >= 64 and frame_count % 10 == 0:
    #     hide.train(64)
    
    # reset            
    if seek.x == hide.x and seek.y == hide.y:
        seeker_episodes.append(seeker_episode_total_reward)
        hider_episodes.append(hider_episode_total_reward)
        
        # flashbulb memory
        # The last thing the agent appended to its memory was the victory frame
        # Duplicate it 5 times so the neural network is forced to study it
        winning_memory = seek.memory[-1]
        for _ in range(5):
            seek.memory.append(winning_memory)
        
        seeker_episode_total_reward = 0.0
        hider_episode_total_reward = 0.0
        episode_count += 1
        step_count = 0
        
        # Randomize positions and update target networks
        randomize_spawns(seek, hide, env)
        seek.update_target_network()
        hide.update_target_network()
        print(f"Gotcha! Episode nb: {episode_count} | Epsilon: {seek.epsilon}")
        
    elif step_count >= MAX_STEPS:
        seeker_episode_total_reward -= 1.0
        hider_episode_total_reward += 1.0
        seeker_episodes.append(seeker_episode_total_reward)
        hider_episodes.append(hider_episode_total_reward)
        seeker_episode_total_reward = 0.0
        hider_episode_total_reward = 0.0
        episode_count += 1
        step_count = 0
        
        # Randomize positions and update target networks
        randomize_spawns(seek, hide, env)
        seek.update_target_network()
        hide.update_target_network()
        print(f"MAX STEPS reached, episode: {episode_count} | Epsilon: {seek.epsilon}")
    
    step_count += 1 
    frame_count += 1
        
    
        
    # event logic
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Generate a unique timestamp 
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save models
            seek.save_model('seeker.keras')
            hide.save_model('hider.keras')
            
            plt.figure(figsize=(15, 10))
            
            # left graph = seeker rewards
            plt.subplot(2, 2, 1)
            plt.plot(seeker_episodes, color='green')
            plt.title('Seeker Total Reward per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            
            # right graph = seeker loss
            plt.subplot(2, 2, 2)
            plt.plot(seek.loss_history, color='red')
            plt.title('Seeker Neural Network Loss')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            
            # left graph = hider rewards
            plt.subplot(2, 2, 3)
            plt.plot(hider_episodes, color='blue')
            plt.title('Hider Total Reward per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            
            # right graph = hider loss
            plt.subplot(2, 2, 4)
            plt.plot(hide.loss_history, color='magenta')
            plt.title('Hider Neural Network Loss')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            
            plt.tight_layout()
            
            # Save the graph with the unique timestamp!
            graph_filename = f'performance_graphs_{timestamp}.png'
            plt.savefig(graph_filename)
            print(f"Model saved. Graphs saved as: {graph_filename}")
            
            pygame.quit()
            sys.exit()
        
        # door logic
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            grid_x = mouse_x // CELL_SIZE
            grid_y = mouse_y // CELL_SIZE
            env.toggle_door(grid_x, grid_y)
    
    # draw the map
    screen.fill(WHITE)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            cell_value = env.grid[y, x]
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            
            if cell_value == 1:
                pygame.draw.rect(screen, BLACK, rect)
            if cell_value == 2:
                pygame.draw.rect(screen, BROWN, rect)
            
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)
            
            
    # draw the agents
    rect_seek = pygame.Rect(seek.x * CELL_SIZE, seek.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, seek.color, rect_seek)
    
    rect_hide = pygame.Rect(hide.x * CELL_SIZE, hide.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, hide.color, rect_hide)
    
    # refresh screen
    pygame.display.flip()
    clock.tick(60)
    
pygame.quit()