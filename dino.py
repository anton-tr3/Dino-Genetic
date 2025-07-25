import pygame
import random
import copy
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

pygame.init()

# Game Constants
FPS = 40
SCREEN_WIDTH = 1100
SCREEN_HEIGHT = 600

GAME_SPEED = 20

RUNNING = [pygame.image.load("assets/Dino/DinoRun1.png"), 
           pygame.image.load("assets/Dino/DinoRun2.png")]

JUMPING = pygame.image.load("assets/Dino/DinoJump.png")

DUCKING = [pygame.image.load("assets/Dino/DinoDuck1.png"), 
           pygame.image.load("assets/Dino/DinoDuck2.png")]

SMALL_CACTUS = [pygame.image.load("assets/Cactus/SmallCactus1.png"),
                pygame.image.load("assets/Cactus/SmallCactus2.png"),
                pygame.image.load("assets/Cactus/SmallCactus3.png")]

LARGE_CACTUS = [pygame.image.load("assets/Cactus/LargeCactus1.png"),
                pygame.image.load("assets/Cactus/LargeCactus2.png"),
                pygame.image.load("assets/Cactus/LargeCactus3.png")]

BIRD = [pygame.image.load("assets/Bird/Bird1.png"),
        pygame.image.load("assets/Bird/Bird2.png")]

CLOUD = pygame.image.load("assets/Other/Cloud.png")

BG = pygame.image.load("assets/Other/Track.png")

# Genetic algorithm constants
POPULATION_SIZE = 300
PARENTS_PER_GENERATION = 50
GENERATIONS = 150
MUTATION_RATE = 0.2
MUTATION_STRENGTH = 1
ELITE_COUNT = 10

class Dinosaur:
    # Define the coordinates of the player
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5

    # Initial properties of the dino
    def __init__(self):
        # Define the images for all 3 actions
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        # Define the initial movement
        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False
        
        # Index of step to control animations
        self.step_index = 0

        # Initial jump velocity
        self.jump_vel = self.JUMP_VEL

        # Initial image of the dino (running)
        self.image = self.run_img[0]

        # Hitbox of the dino equal in position and size to image sprite
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    # Update on each iteration of the main loop
    def update(self):
        # Run these movement functions when required
        # Mid jump this will continuously run so you can't input anything else
        if self.dino_jump:
            self.jump()
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()

        # Reset index every 10 steps
        if self.step_index >= 10:
            self.step_index = 0

    # When the player is ducking
    def duck(self):
        # Animating duck sprites
        self.image = self.duck_img[self.step_index // 5]

        # Update hitbox
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK

        # Update step index
        self.step_index += 1

    # When the player is running
    def run(self):
        # Rotate between the 2 running sprites to animate the character
        # Step index between 0 to 5 will have the first sprite, 6 to 10 will be the second sprite
        self.image = self.run_img[self.step_index // 5]
        
        # Update hitbox size, and then position
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

        # Update step index
        self.step_index += 1

    # When the player is jumping
    def jump(self):
        # Set jump sprite
        self.image = self.jump_img

        # If we are in the jumping state
        if self.dino_jump:
            # Move the dino up by 4x the velocity
            self.dino_rect.y -= self.jump_vel * 4
            # Decrease velocity (gravity)
            self.jump_vel -= 0.8
        
        # If the velocity has gone below -JUMP_VEL, then we have completed the jump
        if self.jump_vel < - self.JUMP_VEL:
            # Remove jumping state
            self.dino_jump = False
            # Reset jump velocity to +8.5
            self.jump_vel = self.JUMP_VEL
            
            # Reset y position
            self.dino_rect = self.image.get_rect()
            self.dino_rect.x = self.X_POS
            self.dino_rect.y = self.Y_POS

    # Draw images to the screen
    def draw(self, SCREEN):
        # Paste the current sprite (ducking, running or jumping) into the given position
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

class Cloud:
    def __init__(self):
        # Set the initial coordinates of the cloud, the x will spawn it off screen
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)

        # Set cloud image and get the width
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        global GAME_SPEED
        # Move the cloud left by the given speed
        self.x -= GAME_SPEED

        # If it reaches the left side of the screen, reset it back to the right off screen
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        # Draw the cloud on the screen
        SCREEN.blit(self.image, (self.x, self.y))

class Obstacle:
    def __init__(self, image, type):
        # Type is an integer from 0 to 2 representing the different kind of cacti that can spawn
        self.type = type

        # Sprite and hitbox
        self.image = image
        self.rect = self.image[self.type].get_rect()

        # Set position to just on the right of the screen
        self.rect.x = SCREEN_WIDTH + random.uniform(0, 50)

    def update(self, obstacles, env):
        global GAME_SPEED
        # Move left at the game speed
        self.rect.x -= GAME_SPEED

        # If we reach the left of the screen, remove from obstacle list
        if self.rect.x < -self.rect.width:
            obstacles.pop(0)
            env.obstacles_cleared += 1
        
        return obstacles

    def draw(self, SCREEN):
        # Draw onto screen
        SCREEN.blit(self.image[self.type], self.rect)

class SmallCactus(Obstacle):
    def __init__(self, image):
        # Set the type of cacti randomly
        self.type = random.randint(0, 2)

        # Initialise init from obstacle class
        super().__init__(image, self.type)

        # Set y position
        self.rect.y = 325

class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        # A bit higher than the small cacti
        self.rect.y = 300

class Bird(Obstacle):
    def __init__(self, image):
        # Only one type of bird
        self.type = 0
        
        super().__init__(image, self.type)
        
        self.rect.y = 250
        
        # For animations
        self.index = 0

    # Bird has animations, override draw function of the obstacle (made for cacti)
    def draw(self, SCREEN):
        # Reset index
        if self.index >= 9:
            self.index = 0

        # Draw bird
        SCREEN.blit(self.image[self.index // 5], self.rect)
        self.index += 1

class DinoEnvironment:
    # By default render the graphics
    def __init__(self, render = True):
        # Set render variable to be used in other functions
        self.render_enabled = render

        # If render is true, setup the screen
        if render == True:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            self.screen = None
        
        # Set clock
        self.clock = pygame.time.Clock()
        self.reset()

    # Runs at the start of each game
    def reset(self):
        # Define player character
        self.dino = Dinosaur()
        self.cloud = Cloud()

        # Set global variables to their starting values 
        self.obstacles = []
        self.total_score = 0
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        self.obstacles_cleared = 0
        self.done = False

        # Reset game speed and objects cleared
        global GAME_SPEED
        GAME_SPEED = 18

        # Returns y position, y velocity and distance to object
        return self.get_state()
    
    # To actually get the state of the game to pass to the neural net
    def get_state(self):
        # If obstacles exist, take the distance between them
        if self.obstacles != []:
            dist = self.obstacles[0].rect.x - self.dino.dino_rect.x
        # Otherwise set it to the width of the screen
        else:
            dist = SCREEN_WIDTH

        # Return the current y position, game speed, the y position of the next obstacle, x dist to it and also whether the dino is in a jump
        # If no obstacles, return a ground obstacle (y = 325)
        global GAME_SPEED
        is_on_ground = 1 if self.dino.dino_rect.y == 310 else 0
        if len(self.obstacles) == 0:
            return [self.dino.dino_rect.y, GAME_SPEED, 325, dist, is_on_ground]
        else:
            return [self.dino.dino_rect.y, GAME_SPEED, self.obstacles[0].rect.y, dist, is_on_ground]

    def get_reward(self):
        if self.done:
            return 75 * self.obstacles_cleared
        
        reward = 0
        if self.dino.dino_duck == True:
            reward += -1
        elif self.dino.dino_jump == True:
            reward += -4
            
        return reward

    def step(self, action):
        # ? Next frame after colliding with an object
        if self.done == True:
            # Return state, reward, done status
            return self.get_state(), self.get_reward(), True
        
        # ? Add one to score each frame and increase speed if needed
        self.total_score += 1
        # Increase game speed every 100 score
        global GAME_SPEED
        if self.total_score % 100 == 0:
            GAME_SPEED += 1

        # ? Exiting the program
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # ? Change state of dino if actions are passed through, also update dino state
        # If not jumping and jump is pressed
        if action == 1 or self.dino.dino_jump == True:
            self.dino.dino_duck = False
            self.dino.dino_run = False
            self.dino.dino_jump = True
        # If not jumping and duck is pressed
        elif action == 2 and not self.dino.dino_jump:
            self.dino.dino_duck = True
            self.dino.dino_run = False
            self.dino.dino_jump = False
        # If not jumping or ducking
        elif not (self.dino.dino_jump or action == 1 or action == 2):
            self.dino.dino_duck = False
            self.dino.dino_run = True
            self.dino.dino_jump = False

        self.dino.update()

        # ? Spawning obstacles
        # If there are no obstacles or if the last object has moved enough, small chance to spawn another
        if len(self.obstacles) == 0 or (self.obstacles[-1].rect.x < SCREEN_WIDTH - 700 and random.randint(0, 100) < 2):
            # Make a new one randomly
            if random.randint(0, 1) == 0:
                self.obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 1) == 0:
                self.obstacles.append(LargeCactus(LARGE_CACTUS))
            else:
                self.obstacles.append(Bird(BIRD))

        for obstacle in self.obstacles:
            self.obstacles = obstacle.update(self.obstacles, self)

            # If we collide with an obstacle
            if self.dino.dino_rect.colliderect(obstacle.rect):
                self.done = True
                break
        
        # ? Render all images and objects if needed
        if self.render_enabled == True:
            self.render()

        # ? Return state, reward and done status
        return self.get_state(), self.get_reward(), self.done
    
    # Rendering everything if required
    def render(self):
        # ? Fill screen with white
        self.screen.fill((255, 255, 255))

        # ? Draw the player onto the screen
        self.dino.draw(self.screen)

        # ? Draw in all obstacles
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        # ? Draw scrolling clouds
        self.cloud.draw(self.screen)
        self.cloud.update()

        # ? Draw scrolling background
        # Draw the background, and another copy on the right of the screen
        image_width = BG.get_width()
        self.screen.blit(BG, (self.x_pos_bg, self.y_pos_bg))
        self.screen.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))

        # If the current background moves off the left of the screen, spawn another on the right of the screen
        if self.x_pos_bg <= -image_width:
            self.screen.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))
            # Reset position of background
            self.x_pos_bg = 0
        
        global GAME_SPEED
        self.x_pos_bg -= GAME_SPEED

        # ? Draw score
        # Create text to display score and set the position to the top right of the screen
        pygame.font.init()
        font = pygame.font.Font('assets/PressStart2P.ttf', 20)
        text = font.render(
            "Score: " + str(self.total_score),
            True, (0, 0, 0)
        )
        textRect = text.get_rect()
        textRect.center = (960, 40)
        self.screen.blit(text, textRect)

        # ? Other rendering commands
        # Control the FPS of the game
        self.clock.tick(FPS)
        # Update display
        pygame.display.update()

    # Close environment
    def close(self):
        pygame.quit()

# Turn the dino y position into a value from 0 to 1
def normalise_player_height(player_y_pos):
    # Highest y value will be when the dino is on the floor. This will correspond to a 0 height in the new scale
    floor_y = 310

    # Lowest y value occurs at max jump height. This will be a 1 on the new scale
    max_player_height = 112

    return abs((player_y_pos - floor_y) / (max_player_height - floor_y))

# y_position of the obstacle can be 2 values: cacti (small cacti = 325, large cacti = 300), or bird = 250, we will convert this to 0, 1 and 2 respectively
def obstacle_pos_to_state(obstacle_y_pos):
    if obstacle_y_pos == 325 or obstacle_y_pos == 300:
        return [1, 0] # cacti
    else:
        return [0, 1] # bird

# Turn the distance to an obstacle into a value from 0 to 1
def normalise_obstacle_distance(obstacle_distance):
    total_dist = SCREEN_WIDTH

    # Fraction of distance to the screen width (0 is right side, 1 is left side of the screen)
    # Don't allow distance to go below 0 (when the object is to the left of the dino and going off screen)
    return max(0, (obstacle_distance / total_dist))

# Turn the game speed into a value from 0 to 1 
def normalise_game_speed(GAME_SPEED):
    # Max speed before the game becomes impossible
    starting_speed = 20
    max_speed = 70

    return (GAME_SPEED - starting_speed) / (max_speed - starting_speed)

# Neural network controlling the dino's actions
class DinoNN(nn.Module):
    def __init__(self):
        super().__init__()
        # We will have 2 fully connected layers and 1 output layer
        self.fc1 = nn.Linear(6, 32)    # 4 inputs (y_pos, GAME_SPEED, 2 types of obstacles, obstacle_dist)
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 3) # 3 outputs (0 = nothing, 1 = jump, 2 = duck)

    def forward(self, x):
        # ReLU activation for the first 2 layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Get output
        x = self.output(x)

        # Convert output into probabilities using softmax to determine action taken
        return F.softmax(x, dim = -1)

# For each parameter, take weights and biases randomly from each parent
def crossover(parent1, parent2):
    # Create a complete copy of the parent1 neural network
    child = copy.deepcopy(parent1)

    for par1_param, par2_param, child_param in zip(parent1.parameters(), parent2.parameters(), child.parameters()):
        # Create a binary mask of input size equal to the current parameters
        bin_mask = torch.rand_like(par1_param) < 0.5

        # Since bin_mask contains 0s and 1s, we can multiply to give the random parameters from both parents
        # This is repeated on the child for each parameter
        child_param.data = (bin_mask * par1_param.data) + (~(bin_mask) * par2_param.data)
    
    return child

# Mutate parameters randomly
def mutate(agent):
    for param in agent.parameters():
        # Binary mask
        bin_mask = torch.rand_like(param) < MUTATION_RATE
        
        # Noise to mutate the parameters
        noise = torch.randn_like(param) * MUTATION_STRENGTH

        # This will mutate the values of the parameter with a 1 in bin_mask by the noise
        param.data += bin_mask * noise

    return agent

# Roulette wheel selection based on fitness score
def selection(agent_population, fitness_scores):
    fitness_scores = [max(0, score) for score in fitness_scores]
    total_score = sum(fitness_scores)
    
    selected = []

    for _ in range(PARENTS_PER_GENERATION):
        # Pick a random float between 0 and the total fitness
        pick = random.uniform(0, total_score)

        # Keep a count of the total accumulated score, once we go over the given pick, we have our person
        total_score = 0
        for agent, score in zip(agent_population, fitness_scores):
            total_score += score

            if total_score >= pick:
                selected.append(agent)
                break

    return selected

# Given a list of parents, generate a new generation by crossing over and mutating
def evolution(parents):
    new_population = []

    while len(new_population) < POPULATION_SIZE:
        parent1, parent2 = random.sample(parents, 2)

        child = crossover(parent1, parent2)
        child = mutate(child)

        new_population.append(child)

    return new_population

# Given a population, run all agents in parallel and return a list of their fitness scores
def simulate_population(population, render = False):
    # All of the environments, states, scores and finished status of population
    environments = [DinoEnvironment(render = render) for _ in population]
    states = [env.reset() for env in environments]
    fitness_scores = [0 for _ in range(len(population))]
    finished = [False for _ in range(len((population)))]

    # Loop until all agents are finished
    while all(finished) == False:
        
        # Loop through each agent in the population
        for index, agent in enumerate(population):
            if finished[index] == True:
                continue

            # Feed input data as a tensor into neural network
            y_pos, GAME_SPEED, obstacle_y_pos, obstacle_dist, is_on_ground = states[index]

            input_tensor = torch.tensor([normalise_player_height(y_pos),
                                        normalise_game_speed(GAME_SPEED),
                                        normalise_obstacle_distance(obstacle_dist),
                                        is_on_ground] +
                                        obstacle_pos_to_state(obstacle_y_pos), 
                                        dtype=torch.float32)

            # Gradient calculation not required 
            with torch.no_grad():
                net_output = agent(input_tensor) # list of 3 items and their probabilities
                action = torch.argmax(net_output).item()  # Get index of largest probability, and turn it into an integer

            # Advance to the next step, updating the state, score and status of the current agent
            state, reward, done = environments[index].step(action)
            
            states[index] = state
            finished[index] = done
            fitness_scores[index] += reward
    
    return fitness_scores

# Run the algorithm fully
def genetic_algorithm():
    population = [DinoNN() for _ in range(POPULATION_SIZE)]

    best_score = -math.inf
    best_avg_score = -math.inf

    for generation in range(GENERATIONS):
        # Get the score of each agent in the generation
        fitness_scores = simulate_population(population)
        
        for index, score in enumerate(fitness_scores):
            if score > best_score:
                best_score = score
                torch.save(population[index].state_dict(), 'dino_top_agent.pth')
                print(f'New Best: {best_score}')

        
        avg_score = np.mean(fitness_scores)
        
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            population_copy = population[:]
            fitness_copy = fitness_scores[:]

            torch.save(population_copy[fitness_copy.index(max(fitness_copy))].state_dict(), 'dino_1_avg_agent.pth')
            fitness_copy.pop(fitness_copy.index(max(fitness_copy)))
            
            torch.save(population_copy[fitness_copy.index(max(fitness_copy))].state_dict(), 'dino_2_avg_agent.pth')
            fitness_copy.pop(fitness_copy.index(max(fitness_copy)))

            torch.save(population_copy[fitness_copy.index(max(fitness_copy))].state_dict(), 'dino_3_avg_agent.pth')
            fitness_copy.pop(fitness_copy.index(max(fitness_copy)))

            print(f'New Best Avg: {best_avg_score}')


        print(f'Average score in generation {generation}: {np.mean(fitness_scores)}')
        #print(f'Max score in generation {generation}: {max(fitness_scores)}')

        # Get the top N agents to keep for the next population
        sorted_agents = [agent for _, agent in sorted(zip(fitness_scores, population), key=lambda p: p[0], reverse=True)]
        elites = sorted_agents[:ELITE_COUNT]

        # Generate new generation
        parents = selection(population, fitness_scores)
        new_population = evolution(parents)

        # Add elites in
        population = elites + new_population[:POPULATION_SIZE - ELITE_COUNT]


        # Run the game using a single agent

def simulate_agent(model_path):
    env = DinoEnvironment(render = True)
    state = env.reset()

    # Load model from file
    model = DinoNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    done = False

    # Play the game using the model
    while done == False:
        y_pos, GAME_SPEED, obstacle_y_pos, obstacle_dist, is_on_ground = state

        input_tensor = torch.tensor([normalise_player_height(y_pos),
                                    normalise_game_speed(GAME_SPEED),
                                    normalise_obstacle_distance(obstacle_dist),
                                    is_on_ground] +
                                    obstacle_pos_to_state(obstacle_y_pos))

        with torch.no_grad():
            net_output = model(input_tensor)
            action = torch.argmax(net_output).item()

        state, reward, done = env.step(action)

genetic_algorithm()
simulate_agent('dino_top_agent.pth')
simulate_agent('dino_1_avg_agent.pth')
simulate_agent('dino_2_avg_agent.pth')
simulate_agent('dino_3_avg_agent.pth')