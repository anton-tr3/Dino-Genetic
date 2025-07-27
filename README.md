# 🦖 Chrome Dinosaur AI: Genetic Algorithm Agent
This project trains an AI to play a recreation of the Chrome Dinosaur game using a genetic algorithm that evolves a neural network to learn the game environment. 

## Features

### Default Parameters

## Usage
### 1. Clone repo
 ```
 git clone https://github.com/anton-tr3/dino-genetic-ai.git
 ```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Run genetic algorithm
Train the agents with the given parameters. Fitness of each generation will be printed on the console. The top scoring agent along with the top 3 agents of the best performing generation will be saved in `/output_agents`.
```
python dino.py -genetic
```

### 4. Replay saved agents
Plays the simulation using a pre-trained agent `[agent].pth` generated through running the genetic algorithm
```
python dino.py --play [path_to_agent]
```
