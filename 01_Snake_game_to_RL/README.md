# Snake Game

After the use of OOP to have the snake working here are the phases we will use to impliment the Neural Network + Genetic Algorithm


#### How to run it:
while in the Nearal_Network folder

```python
python watch_trained_snake.py
```


### Phase 1 — State Representation

What does the snake "see"? You'll design a vector of numbers that describes the world. This is the input to your neural network. Every ML system starts with this question: what information does the model need?

**Why state representation matters:**

A neural network can't look at your screen. It can only receive numbers.

So the question to ask is;

*"How do I describe the world as a vector of numbers?"* Every project — image classifiers, language models, trading bots — starts here.

For the snake, the state needs to answer: *"What does the snake need to know to make a smart decision right now?"*

We'll start with just **3 pieces of information** — danger in three directions relative to the snake's current heading:
[danger_straight, danger_left, danger_right]

### Phase 2 — The Neural Network

A tiny network: input layer → hidden layer → output layer. You'll build this with nothing but Python lists and basic math. You'll understand what a weight is, what a bias is, and what an activation function does — because you'll write them yourself.

<!-- ## Food Direction — Giving the Snake Goals -->
#### Food Direction — Giving the Snake Goals

We have the state vector is `[danger_straight, danger_left, danger_right]`. The snake can sense death but it has no idea where food is. A snake that can only avoid danger will just wander randomly — it needs a **goal signal**.

This is a fundamental ML concept called **feature engineering** — deciding what information your model needs to make good decisions. Danger alone is defensive. Food direction is offensive. Together they give the agent enough information to actually learn a strategy.

#### What We're Adding

Four more values describing where the food is **relative to the head**:

```python
[food_left, food_right, food_up, food_down]
```

These are simple booleans — is the food somewhere to my left? Somewhere above me? Not how far, just which direction.

Your final state vector becomes:

```
[danger_straight, danger_left, danger_right, food_left, food_right, food_up, food_down]
```

Seven numbers. That's the complete input your neural network will eventually receive.

#### The Math

This one is simpler than the danger calculation — no heading-relative thinking needed. Food direction is **absolute**, not relative:

```
food is to the left  → food_x < head_x
food is to the right → food_x > head_x
food is above        → food_y > head_y
food is below        → food_y < head_y
```

That's it. Pure comparison operators.

<!-- ## Your Task -->

You already have `food` passed into `get_state(self, food)` — you just haven't used it yet.

The food object has `.xcor()` and `.ycor()` methods, exactly like the snake's head.

**Add the food direction calculation to `get_state()`** and extend the return value from 3 numbers to 7.

Two questions to guide you:

- Can food be to the left **and** above at the same time? What does that mean for your booleans?
- What happens when the food is at the exact same x or y coordinate as the head? Does that matter?

Try it, share your attempt, and we'll refine together.

### Phase 3: The Neural Network — Building a Brain from Scratch

Instead of backpropagation (which requires calculus), you'll evolve a population of snakes. The ones that survive longest pass their "genes" (network weights) to the next generation, with small mutations. This is how nature trains neural networks.

#### The Architecture

The network will have **three layers**:

```
Input Layer (7 neurons)  →  Hidden Layer (12 neurons)  →  Output Layer (3 neurons)
     ↓                              ↓                             ↓
[danger + food info]          [pattern detection]          [straight, left, right]
```

**Input layer:** The 7 values from `get_state()` — danger and food direction.

**Hidden layer:** 12 neurons that learn to detect patterns. Why 12? It's a reasonable starting point — enough to learn complex behavior, not so many it becomes slow. Can experiment later.

**Output layer:** 3 neurons, one for each possible move. The highest value wins — that's the direction the snake chooses.

#### The Math Concept: What IS a Neural Network?

At its core, a neural network is just **repeated matrix multiplication and squashing**.

Each connection between layers has a **weight** — a number that gets multiplied by the input. Each neuron has a **bias** — a number that gets added after multiplication. Then you apply an **activation function** to squash the result into a useful range.

Here's one neuron's calculation:

```
output = activation(sum(inputs * weights) + bias)
```

Do that for every neuron in every layer, and you've got a neural network.

#### The Activation Function: Tanh

You need a function that takes any number and squashes it into a fixed range. We'll use **tanh** (hyperbolic tangent), which maps any input to between -1 and 1:

```
tanh(0)    = 0
tanh(100)  = 0.999...  (approaching 1)
tanh(-100) = -0.999... (approaching -1)
```

Why tanh instead of sigmoid or ReLU? For this simple project, tanh works well because it's centered at zero (outputs can be negative) and it's smooth. Don't overthink it — the activation function choice matters less than understanding *why* you need one.

Python gives you `math.tanh()` for free.

#### The Network Class

`01_Snake_game_to_RL/Neural_Network/neural_network.py`

Here's the skeleton. You can go through and try to undestnad the logic before implimenting; read and understand the structure:

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=7, hidden_size=12, output_size=3):
        """
        Initialize the network with random weights and biases.
        
        Architecture:
        - Input layer: input_size neurons (7 for our state)
        - Hidden layer: hidden_size neurons (12)
        - Output layer: output_size neurons (3 for directions)
        """
        # Weights between input and hidden layer
        # Shape: (input_size, hidden_size) - why this shape?
        self.weights_input_hidden = ???
        
        # Biases for hidden layer
        # Shape: (hidden_size,) - one bias per hidden neuron
        self.bias_hidden = ???
        
        # Weights between hidden and output layer
        # Shape: (hidden_size, output_size)
        self.weights_hidden_output = ???
        
        # Biases for output layer
        # Shape: (output_size,)
        self.bias_output = ???
    
    def forward(self, inputs):
        """
        Forward propagation: inputs → hidden → output
        
        This is how the network "thinks" - takes state, produces action.
        """
        # Step 1: Input to hidden layer
        # hidden = tanh(inputs * weights_input_hidden + bias_hidden)
        
        # Step 2: Hidden to output layer  
        # output = tanh(hidden * weights_hidden_output + bias_output)
        
        # Step 3: Return the output
        pass
```

#### Questions to Think About

**Before you implement anything**, answer these conceptually:

1. **Why random weights?** Why not start all weights at 0 or 1?

    <details>
    <summary>Click to expand/collapse the hidden content</summary>

        If you start all weights at **the same value** (like all zeros or all ones), every neuron in a layer will learn **the exact same thing**. They'll all update identically during training because they all start identically. You've essentially built a network with just one neuron repeated 12 times — a waste.

        Random weights break this symmetry. Each neuron starts with different weights, so each learns to detect different patterns. Some might learn "food is close and no danger = good", others might learn "tight corner ahead = bad". Diversity of weights = diversity of learned patterns.

        **Key insight:** Randomness isn't about finding the best value directly — it's about giving each neuron a different starting point so they specialize differently.

    </details>

2. **What does the shape `(7, 12)` mean?** If you multiply a vector of 7 numbers by a matrix of shape `(7, 12)`, what shape is the result?

    <details>
    <summary>Click to expand/collapse the hidden content</summary>

                12 columns
            ┌─────────────┐
        7 rows │   weights   │
            └─────────────┘

        **What happens when you multiply?**

        inputs (shape: 7)  ×  weights (shape: 7×12)  =  result (shape: 12)

        Think of it row-by-row:

        - Input neuron 1 connects to all 12 hidden neurons → that's row 1 of the weight matrix
        - Input neuron 2 connects to all 12 hidden neurons → row 2
        - ...and so on

        Each of the 12 columns represents the incoming weights **to one hidden neuron** from all 7 inputs.

        The result is 12 numbers — one for each hidden neuron.

        **Visual:**

        [a, b, c, d, e, f, g]  ×  [7×12 matrix]  =  [h₁, h₂, h₃, ..., h₁₂]
            7 inputs              weights           12 hidden neurons

    </details>


3. **Why tanh twice?** Why apply activation after the hidden layer AND the output layer?
    <details>
    <summary>Click to expand/collapse the hidden content</summary>

    ```
    **After the hidden layer:** Tanh squashes the weighted sum into the range `[-1, 1]`. This keeps the values bounded and adds non-linearity. Without it, stacking layers would just be repeated linear transformations — which is mathematically equivalent to a single layer. The activation function is what gives neural networks the power to learn complex, non-linear patterns.

    **After the output layer:** Tanh again squashes the output into `[-1, 1]`. This makes the outputs comparable — you can't compare "neuron 1 says 0.5" vs "neuron 2 says 10000" meaningfully. Squashing puts them on the same scale.

    **Key insight:** Activation functions don't just squash — they introduce non-linearity, which is what makes deep learning possible.

    ```
    
   
    </details>


4. **How does this produce a decision?** The output is 3 numbers. How do you turn `[0.3, -0.1, 0.8]` into a choice of "go right"?

    <details>
    <summary>Click to expand/collapse the hidden content</summary>

        **Tanh outputs a range, not just -1, 0, 1:**

        tanh(-5) = -0.9999
        tanh(-1) = -0.76
        tanh(0)  =  0.0
        tanh(1)  =  0.76
        tanh(5)  =  0.9999

        It's a smooth curve, not discrete jumps.

        **Your example: `[0.3, -0.1, 0.8]`**

        These are three output neurons — one for each direction:

        - Neuron 0 (straight): 0.3
        - Neuron 1 (left): -0.1
        - Neuron 2 (right): 0.8

        **The decision rule:** Pick the neuron with the **highest value**.

        In this case, neuron 2 has the highest value (0.8), so the snake chooses **right**.

        It doesn't matter if the values are close to -1 or 1 — what matters is which one is **bigger than the others**. The network is saying "I'm most confident about going right."
    </details>


<!-- ### Phase 4 — Visualization -->

### Phase 4: Genetic Algorithm — Teaching Through Evolution

This is where the network learns to play. No calculus, no backpropagation — just survival of the fittest.

#### The Core Concept: Evolution as an Optimization Algorithm

In nature, organisms with better traits survive longer and pass their genes to offspring. Over generations, the population gets better at surviving.

We're going to do the exact same thing with neural networks:

1. Create a **population** of snakes, each with a different random brain (neural network)
2. Let them all play the game until they die
3. The snakes that survive longest are the "fittest" — their brains made good decisions
4. **Kill the weak**, **keep the strong**
5. **Breed** the survivors — create new snakes by mixing the best brains with small random mutations
6. Repeat for many generations

Over time, the population evolves better and better gameplay.

<!-- ## Why This Works for Your Project

**No gradient calculation needed.** You don't need to understand derivatives or backpropagation. The "training signal" is just: did the snake die quickly or survive long?

**Intuitive to understand.** You can literally watch snakes getting smarter generation by generation.

**Directly maps to the game.** Fitness = score or survival time. Simple.

--- -->

#### The Algorithm Structure

Here's the flow to implement:

```
Generation 1:
├─ Create 50 snakes with random brains
├─ Each snake plays until it dies
├─ Record fitness (score or steps survived)
├─ Rank by fitness
└─ Select top 10 as "parents"

Generation 2:
├─ Keep the top 10 (elitism)
├─ Create 40 new snakes by:
│  ├─ Pick 2 random parents
│  ├─ Create child by mixing their weights (crossover)
│  └─ Add small random changes (mutation)
└─ Repeat the play → rank → select cycle

Generation 3, 4, 5, ... 100:
└─ Same process, snakes get better each time
```

---

#### The Key Operations

##### 1. **Fitness Function**
How do you measure a snake's performance? Two common options:

**Option A:** Score (how much food it ate)  
**Option B:** Steps survived (how long it lived)  

We'll use **score** as primary, **steps** as tiebreaker. A snake that eats 5 foods is better than one that eats 3, even if the second one lived longer by wandering.

#### 2. **Selection**
Keep the top performers. Simple ranking.

#### 3. **Crossover** (Breeding)
Mix two parent networks to create a child:

```
Parent 1 weights: [0.5, -0.3,  0.8, ...]
Parent 2 weights: [0.1,  0.9, -0.2, ...]
                       ↓ mix ↓
Child weights:    [0.5,  0.9,  0.8, ...]  ← some from each parent
```

#### 4. **Mutation**
After creating a child, randomly tweak some weights:

```
Original: 0.5 → mutate → 0.53  (small random change)
```

This explores new strategies. Without mutation, you'd just keep recombining the same genes forever.


#### What to Build

A new class: `GeneticTrainer`

It will handle:
- Managing a population of neural networks
- Running each snake through the game
- Calculating fitness scores
- Selecting, breeding, and mutating networks
- Tracking progress across generations

Here's the skeleton:

```python
from snake import Snake
from food import Food
from neural_network import NeuralNetwork
import numpy as np

def play_game(network, max_steps=1000):
    """
    Play one game using the given neural network.
    Returns fitness score.
    
    max_steps: Stop the game if snake survives this long without eating
               (prevents infinite wandering)
    """
    snake = Snake()
    food = Food()
    
    score = 0
    steps = 0
    steps_since_food = 0
    
    while steps < max_steps:
        # Get current state
        state = snake.get_state(food)
        
        # Network decides which direction
        output = network.forward(state)
        action = np.argmax(output)  # 0=straight, 1=left, 2=right
        
        # TODO: Convert action to actual snake movement
        # How do you turn "go left" into calling snake.left()?
        
        # Move the snake
        snake.move()
        steps += 1
        steps_since_food += 1
        
        # Check collision with food
        if snake.head.distance(food) < 15:
            food.refresh()
            snake.extend()
            score += 1
            steps_since_food = 0
        
        # Check collision with wall
        if (snake.head.xcor() > 280 or snake.head.xcor() < -280 or 
            snake.head.ycor() > 280 or snake.head.ycor() < -280):
            break
        
        # Check collision with tail
        for segment in snake.segments[1:]:
            if snake.head.distance(segment) < 10:
                break
        
        # Timeout if no food eaten recently (prevent infinite loops)
        if steps_since_food > 100:
            break
    
    # Fitness = score is priority, steps as tiebreaker
    return score * 1000 + steps
```

### Phase 5: The Genetic Algorithm Trainer

This is the final piece we are going to build a system that evolves a population of snakes over many generations.



#### The Core Loop

Here's the structure:

```
Initialize Generation 0:
├─ Create 50 snakes with random brains
└─ This is your starting population

For each generation (1 to 100):
├─ Evaluation Phase:
│  ├─ Each snake plays the game (headless, fast)
│  ├─ Record its fitness score
│  └─ Rank all snakes by fitness
│
├─ Selection Phase:
│  └─ Keep the top 10 performers (elitism)
│
└─ Breeding Phase:
   ├─ Create 40 new children by:
   │  ├─ Pick 2 random parents from top 10
   │  ├─ Crossover: mix their weights
   │  └─ Mutation: add small random noise
   └─ New population = 10 parents + 40 children = 50 total
```

#### The Key Operations

##### 1. **Crossover (Breeding)**

Take two parent networks and create a child by mixing their weights:

```python
# Parent 1's weights for one layer
parent1_weights = [0.5, -0.3, 0.8, 0.1, ...]

# Parent 2's weights for same layer  
parent2_weights = [0.2, 0.9, -0.4, 0.7, ...]

# Child: randomly take each weight from either parent
child_weights = [0.5,  0.9,  0.8,  0.7, ...]  # mix of both
                  ↑P1   ↑P2   ↑P1   ↑P2
```

This is like genetic inheritance — some traits from mom, some from dad.

##### 2. **Mutation**

After creating a child, randomly tweak some weights:

```python
# Before mutation
weight = 0.5

# After mutation (add small random noise)
weight = 0.5 + random.uniform(-0.1, 0.1) = 0.53
```

**Mutation rate:** What percentage of weights get mutated? Start with 10% (1 in 10 weights gets tweaked).

**Mutation strength:** How big is the change? Start with ±0.1 (small tweaks, not huge jumps).

##### 3. **Elitism**

Always keep the best performers unchanged in the next generation. This ensures you never lose your best solution — you can only improve or stay the same.

#### What to Build

**Create a new file:** `genetic_trainer.py`

Here's the skeleton:

```python
import numpy as np
from neural_network import NeuralNetwork
from ai_player import play_game
import random

class GeneticTrainer:
    def __init__(self, population_size=50, elite_size=10, mutation_rate=0.1, mutation_strength=0.1):
        """
        population_size: Total number of snakes per generation
        elite_size: How many top performers to keep unchanged
        mutation_rate: Probability a weight gets mutated (0.1 = 10%)
        mutation_strength: How much to change mutated weights (±0.1)
        """
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        
        # Create initial population
        self.population = [NeuralNetwork() for _ in range(population_size)]
        
        # Track statistics
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def evaluate_population(self):
        """
        Run each network through the game and record fitness.
        Returns list of (network, fitness) tuples sorted by fitness.
        """
        pass  # TODO
    
    def crossover(self, parent1, parent2):
        """
        Create a child network by mixing weights from two parents.
        Returns a new NeuralNetwork.
        """
        pass  # TODO
    
    def mutate(self, network):
        """
        Randomly tweak some weights in the network (in-place).
        """
        pass  # TODO
    
    def evolve_generation(self):
        """
        One full generation cycle: evaluate, select, breed, mutate.
        """
        pass  # TODO
    
    def train(self, generations=100):
        """
        Run the evolution for multiple generations.
        Print progress and track statistics.
        """
        for gen in range(generations):
            self.evolve_generation()
            
            # Print progress every 10 generations
            if gen % 10 == 0:
                print(f"Generation {gen}: Best={self.best_fitness_history[-1]}, Avg={self.avg_fitness_history[-1]:.0f}")
```
