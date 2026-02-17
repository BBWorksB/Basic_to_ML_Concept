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
        fitness_list = []
        for network in self.population:
            fitness = play_game(network)
            fitness_list.append((network, fitness))
        return sorted(fitness_list, key=lambda x: x[1], reverse=True)
    
    def crossover(self, parent1, parent2):
        """
        Create a child network by mixing weights from two parents.
        Returns a new NeuralNetwork.
        """
        child = NeuralNetwork()

        # Mix weights_input_hidden
        mask = np.random.rand(*parent1.weights_input_hidden.shape) > 0.5
        child.weights_input_hidden = np.where(mask, parent1.weights_input_hidden, parent2.weights_input_hidden)
        
        # Mix bias_hidden
        mask = np.random.rand(*parent1.bias_hidden.shape) > 0.5
        child.bias_hidden = np.where(mask, parent1.bias_hidden, parent2.bias_hidden)
        
        # TODO: Do the same for weights_hidden_output and bias_output
        mask = np.random.rand(*parent1.weights_hidden_output.shape) > 0.5
        child.weights_hidden_output = np.where(mask, parent1.weights_hidden_output, parent2.weights_hidden_output)
        
        mask = np.random.rand(*parent1.bias_output.shape) > 0.5
        child.bias_output = np.where(mask, parent1.bias_output, parent2.bias_output)
        
        return child
    
    def mutate(self, network):
        """
        Randomly tweak some weights in the network (in-place).
        """
        # Mutate weights_input_hidden
        mutation_mask = np.random.rand(*network.weights_input_hidden.shape) < self.mutation_rate
        mutations = np.random.uniform(-self.mutation_strength, self.mutation_strength, 
                                    network.weights_input_hidden.shape)
        network.weights_input_hidden += mutation_mask * mutations
        
        # Mutate bias_hidden
        mutation_mask = np.random.rand(*network.bias_hidden.shape) < self.mutation_rate
        mutations = np.random.uniform(-self.mutation_strength, self.mutation_strength, 
                                    network.bias_hidden.shape)
        network.bias_hidden += mutation_mask * mutations
        
        # Mutate weights_hidden_output
        mutation_mask = np.random.rand(*network.weights_hidden_output.shape) < self.mutation_rate
        mutations = np.random.uniform(-self.mutation_strength, self.mutation_strength, 
                                    network.weights_hidden_output.shape)
        network.weights_hidden_output += mutation_mask * mutations
        
        # Mutate bias_output
        mutation_mask = np.random.rand(*network.bias_output.shape) < self.mutation_rate
        mutations = np.random.uniform(-self.mutation_strength, self.mutation_strength, 
                                    network.bias_output.shape)
        network.bias_output += mutation_mask * mutations
    
    def evolve_generation(self):
        """
        One full generation cycle: evaluate, select, breed, mutate.
        """
        # Evaluate current population
        fitness_list = self.evaluate_population()
        
        # Track statistics
        self.best_fitness_history.append(fitness_list[0][1])
        self.avg_fitness_history.append(np.mean([f for _, f in fitness_list]))
        
        # Select elite (top performers)
        elite = [network for network, _ in fitness_list[:self.elite_size]]
        
        # Create new population by breeding elite members
        new_population = elite.copy()
        
        # Fill the rest of the population with offspring from elite members
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(elite, 2)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
    
    def train(self, generations=100):
        """
        Run the evolution for multiple generations.
        Print progress and track statistics.
        """
        for gen in range(generations):
            self.evolve_generation()
            
            # Print progress every 10 generations
            if gen % 10 == 0:
                print(f"Generation {gen}: \
                      Best={self.best_fitness_history[-1]}, Avg={self.avg_fitness_history[-1]:.0f}")



if __name__ == "__main__":
    print("=== FINAL TRAINING RUN ===")
    trainer = GeneticTrainer(
        population_size=100,
        elite_size=15,
        mutation_rate=0.12,
        mutation_strength=0.12
    )
    trainer.train(generations=150)
    
    print("\n=== TRAINING COMPLETE ===")
    print(f"Final best fitness: {trainer.best_fitness_history[-1]}")
    print(f"That's approximately {trainer.best_fitness_history[-1]//1000} foods eaten!")
    
    # Save
    best_network = trainer.population[0]
    np.savez('best_snake_brain_v2.npz',
             w_ih=best_network.weights_input_hidden,
             b_h=best_network.bias_hidden,
             w_ho=best_network.weights_hidden_output,
             b_o=best_network.bias_output)
    print("Saved as 'best_snake_brain_v2.npz'")