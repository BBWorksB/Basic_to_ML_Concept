from neural_network import NeuralNetwork
from game_engine import GameEngine
import numpy as np

def play_game(network, max_steps=2000):
    """Play one game using the neural network. Pure Python, no graphics."""
    game = GameEngine()
    
    while not game.game_over and game.steps < max_steps:
        state = game.get_state()
        output = network.forward(state)
        action = np.argmax(output)
        game.step(action)
    
    return game.score * 1000 + game.steps