import pygame
import sys
import random
from settings import *
from environment import Environment
from agent import Agent

# initialization
pygame.init()
screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
pygame.display.set_caption("Multi agent Hide & Seek")
clock = pygame.time.Clock()

env = Environment()
seek = Agent(1, 1, RED)
hide = Agent(GRID_WIDTH - 2, GRID_HEIGHT - 2, BLUE)


# main loop
running = True
frame_count = 0
while running:
    
    # event logic
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        # door logic
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            grid_x = mouse_x // CELL_SIZE
            grid_y = mouse_y // CELL_SIZE
            env.toggle_door(grid_x, grid_y)
        
        # seek logic
        seek.think_and_move(env, hide)
                
        if seek.x == hide.x and seek.y == hide.y:
            seek.reset()
            hide.reset()
            print("Gotcha!")
            
        frame_count += 1
        
        if frame_count % 10 == 0:
            options = [(0, - 1), (0, 1), (1, 0), (-1, 0), (0, 0)]
            dx, dy = random.choice(options)
            hide.move(dx, dy, env)
    
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