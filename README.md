# 🦖 Chrome Dinosaur AI: Genetic Algorithm Agent
This project trains an AI to play a recreation of the Chrome Dinosaur game, using a genetic algorithm that evolves several neural networks to learn the game environment. 

## Implementation

### Genetic Algorithm Components
Selection  
- Roulette wheel selection is applied, where parents are chosen based on their relative fitness. Higher-performing agents are more likely to be selected to be parents.

Crossover  
- For each pair of parents, their neural network parameters are combined using a random binary mask to produce a child (parameters are taken at random from each parent).

Mutation  
- Each child has a chance for small random mutations to their neural network weights to introduce variation.

Elitism  
- The top-performing agents of each generation (elite count) taken into the next generation unchanged.

### Fitness Function
- Rewards clearing obstacles: +75 per obstacle
- Penalizes jumping: -4 per frame
- Penalizes ducking: -1 per frame

### Default Parameters
|              Parameter | Value |
| ---------------------: | :---- |
|        Population Size | 300   |
| Parents per Generation | 50    |
|            Generations | 150   |
|          Mutation Rate | 0.2   |
|      Mutation Strength | 1     |
|            Elite Count | 10    |

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
