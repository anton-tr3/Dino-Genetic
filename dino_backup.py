import pygame
import random

pygame.init()

# Constants
FPS = 30
SCREEN_WIDTH = 1100
SCREEN_HEIGHT = 600
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

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
    def update(self, userInput):
        # Run these movement functions when required
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        # Mid jump this will continuously run so you can't input anything else
        if self.dino_jump:
            self.jump()

        # Reset index every 10 steps
        if self.step_index >= 10:
            self.step_index = 0

        # If not jumping and jump is pressed
        if userInput[pygame.K_UP] and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        # If not jumping and duck is pressed
        elif userInput[pygame.K_DOWN] and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        # If not jumping or ducking
        elif not (self.dino_jump or userInput[pygame.K_DOWN]):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

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
        # Move the cloud left by the given speed
        self.x -= game_speed

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
        self.rect.x = SCREEN_WIDTH

    def update(self):
        # Move left at the game speed
        self.rect.x -= game_speed

        # If we reach the left of the screen, remove from obstacle list
        if self.rect.x < -self.rect.width:
            obstacles.pop()

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


def main():
    # ? Global variables
    # Speed clouds and background move at, position of backgrounds, score, obstacles list
    global game_speed, x_pos_bg, y_pos_bg, total_score, obstacles
    
    game_speed = 14
    x_pos_bg = 0
    y_pos_bg = 380
    total_score = 0
    obstacles = []

    # ? Classes
    player = Dinosaur()
    cloud = Cloud()

    run = True
    clock = pygame.time.Clock()
    font = pygame.font.Font('freesansbold.ttf', 20)

    def score():
        # Get variables
        global game_speed, total_score
        
        # Increment score by 1 every frame
        total_score += 1

        # Increase game speed every 100 score
        if total_score % 100 == 0:
            game_speed += 1

        # Create text to display score and set the position to the top right of the screen
        text = font.render("Score: " + str(total_score), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        SCREEN.blit(text, textRect)

    def background():
        # Get the global variables
        global x_pos_bg, y_pos_bg

        # Draw the background, and another copy on the right of the screen
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))

        # If the current background moves off the left of the screen, spawn another on the right of the screen
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            # Reset position of background
            x_pos_bg = 0

        x_pos_bg -= game_speed

    while run:
        # If we exit the game window, quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Fill screen with white
        SCREEN.fill((255, 255, 255))
        
        userInput = pygame.key.get_pressed()

        # ? Player Dino
        # Draw the player onto the screen
        player.draw(SCREEN)
        # Update the player's status (duck, run, jump) based on player input
        player.update(userInput)

        # ? Obstacles
        # If there are no obstacles
        if len(obstacles) == 0:
            # Make a new one randomly
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))

            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))

            elif random.randint(0, 2) == 2:
                obstacles.append(Bird(BIRD))

        # Draw obstacles onto the screen
        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update()

            # If we collide with an obstacle
            if player.dino_rect.colliderect(obstacle.rect):
                #pygame.draw.rect(SCREEN, (255, 0, 0), player.dino_rect, 2)
                pygame.time.delay(2000)
                final_score = total_score
                run = False

        # ? Cloud Scrolling
        cloud.draw(SCREEN)
        cloud.update()

        # ? Background Scrolling
        background()

        # ? Score Text
        score()

        # Control the FPS of the game
        clock.tick(FPS)
        # Update display
        pygame.display.update()

main()