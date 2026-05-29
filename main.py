import pygame
import sys
import random
import matplotlib.pyplot as plt
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
# seek.load_model('seeker.keras')
hide = Agent(GRID_WIDTH - 2, GRID_HEIGHT - 2, BLUE, role="hider")
seeker_episode_total_reward = 0.0
seeker_episodes = []
hider_episode_total_reward = 0.0
hider_episodes = []
episode_count = 0
step_count = 0


# main loop
running = True
frame_count = 0
seek.reset(env, hide)
hide.reset(env, seek)
while running:
    
    # seek logic
    seeker_step_reward = seek.think_and_move(env, hide)
    seeker_episode_total_reward += seeker_step_reward
    if len(seek.memory) >= 64 and frame_count % 10 == 0:
        seek.train(64)
            
    # hider logic
    hider_step_reward = hide.think_and_move(env, seek)
    hider_episode_total_reward += hider_step_reward
    if len(hide.memory) >= 64 and frame_count % 10 == 0:
        hide.train(64)
    
    # reset            
    if seek.x == hide.x and seek.y == hide.y:
        seek.reset(env, hide)
        hide.reset(env, seek)
        seeker_episodes.append(seeker_episode_total_reward)
        hider_episodes.append(hider_episode_total_reward)
        seeker_episode_total_reward = 0.0
        hider_episode_total_reward = 0.0
        episode_count += 1
        step_count = 0
        seek.update_target_network()
        hide.update_target_network()
        print(f"Gotcha! Episode nb: {episode_count} | Epsilon: {seek.epsilon}")
    elif step_count >= MAX_STEPS:
        seek.reset(env, hide)
        hide.reset(env, seek)
        seeker_episode_total_reward -= 1.0
        hider_episode_total_reward += 1.0
        seeker_episodes.append(seeker_episode_total_reward)
        hider_episodes.append(hider_episode_total_reward)
        seeker_episode_total_reward = 0.0
        hider_episode_total_reward = 0.0
        episode_count += 1
        step_count = 0
        seek.update_target_network()
        hide.update_target_network()
        print(f"MAX STEPS reached, episode: {episode_count} | Epsilon: {seek.epsilon}")
    
    step_count += 1 
    frame_count += 1
        
    
        
    # event logic
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            seek.save_model('seeker.keras')
            hide.save_model('hider.keras')
            
            plt.figure(figsize=(15, 10))
            
            # left graph = seeker rewards
            plt.subplot(2, 2, 1) # 1 row, 2 cols, 1st graph
            plt.plot(seeker_episodes, color='green')
            plt.title('Seeker Total Reward per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            
            # right graph = seeker loss
            plt.subplot(2, 2, 2) # 1 row, 2 cols, 2nd graph
            plt.plot(seek.loss_history, color='red')
            plt.title('Seeker Neural Network Loss')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            
            # left graph = hider rewards
            plt.subplot(2, 2, 3) # 1 row, 2 cols, 1st graph
            plt.plot(hider_episodes, color='blue')
            plt.title('Hider Total Reward per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            
            # right graph = hider loss
            plt.subplot(2, 2, 4) # 1 row, 2 cols, 2nd graph
            plt.plot(hide.loss_history, color='magenta')
            plt.title('Hider Neural Network Loss')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            
            plt.tight_layout()
            plt.savefig('performance_graphs.png')
            print("Model and graphs saved successfully.")
            
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
    clock.tick(30)
    
pygame.quit()