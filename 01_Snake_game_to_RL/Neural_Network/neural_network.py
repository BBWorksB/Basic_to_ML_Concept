import numpy as np
from snake import Snake
import random

# snake = Snake()

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
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        
        # Biases for hidden layer
        # Shape: (hidden_size,) - one bias per hidden neuron
        self.bias_hidden = np.zeros(hidden_size)
        
        # Weights between hidden and output layer
        # Shape: (hidden_size, output_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        
        # Biases for output layer
        # Shape: (output_size,)
        self.bias_output = np.zeros(output_size)
    
    def forward(self, inputs):
        """
        Forward propagation: inputs → hidden → output
        
        This is how the network "thinks" - takes state, produces action.
        """
        # Step 1: Input to hidden layer
        hidden = np.tanh(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        
        # Step 2: Hidden to output layer  
        output = np.tanh(np.dot(hidden, self.weights_hidden_output) + self.bias_output)
        
        # Step 3: Return the output
        return output


if __name__ == "__main__":
    nn = NeuralNetwork()
    
    # Fake state: danger left, food is right and above
    test_state = [0, 1, 0,  0, 1, 1, 0]
    
    output = nn.forward(test_state)
    print(f"Network output: {output}")
    print(f"Chosen direction: {['straight', 'left', 'right'][np.argmax(output)]}")