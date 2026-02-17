from turtle import Screen, Turtle
from neural_network import NeuralNetwork
from game_engine import GameEngine
import numpy as np
import time

def load_trained_network(filepath='best_snake_brain_v2.npz'):
    """Load a saved network from file."""
    network = NeuralNetwork()
    data = np.load(filepath)
    network.weights_input_hidden = data['w_ih']
    network.bias_hidden = data['b_h']
    network.weights_hidden_output = data['w_ho']
    network.bias_output = data['b_o']
    return network

def play_visual(network):
    """Watch the trained snake play with graphics."""
    screen = Screen()
    screen.setup(width=600, height=600)
    screen.bgcolor("black")
    screen.title("Trained AI Snake - 50 Foods!")
    screen.tracer(0)
    
    # Create visual snake using turtle
    snake_segments = []
    head = Turtle("square")
    head.color("white")
    head.penup()
    head.goto(0, 0)
    snake_segments.append(head)
    
    # Create visual food
    food = Turtle("circle")
    food.color("blue")
    food.penup()
    food.shapesize(0.5, 0.5)
    
    # Create score display
    score_display = Turtle()
    score_display.hideturtle()
    score_display.color("white")
    score_display.penup()
    score_display.goto(0, 270)
    
    # Use game engine for logic
    game = GameEngine()
    
    def update_visuals():
        """Sync turtle graphics with game engine state."""
        # Update snake segments
        while len(snake_segments) < len(game.snake_positions):
            segment = Turtle("square")
            segment.color("white")
            segment.penup()
            snake_segments.append(segment)
        
        for i, pos in enumerate(game.snake_positions):
            snake_segments[i].goto(pos[0], pos[1])
        
        # Update food
        food.goto(game.food_pos[0], game.food_pos[1])
        
        # Update score
        score_display.clear()
        score_display.write(f"Score: {game.score}", align="center", font=("Arial", 24, "normal"))
    
    # Game loop
    while not game.game_over and game.steps < 2000:
        state = game.get_state()
        output = network.forward(state)
        action = np.argmax(output)
        
        game.step(action)
        update_visuals()
        
        screen.update()
        time.sleep(0.03)  # Faster playback
    
    # Final message
    result = Turtle()
    result.hideturtle()
    result.color("white")
    result.penup()
    result.goto(0, 0)
    result.write(f"GAME OVER\nFinal Score: {game.score}", align="center", font=("Arial", 32, "bold"))
    
    screen.exitonclick()

if __name__ == "__main__":
    print("Loading trained snake brain (v2)...")
    network = load_trained_network()
    print("Watching the AI play...\n")
    print("This snake was trained to eat 50 foods!")
    print("Watch how it navigates toward food and avoids walls/tail.\n")
    play_visual(network)