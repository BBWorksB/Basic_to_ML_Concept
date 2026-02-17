from turtle import Screen
from snake import Snake
from food import Food
from neural_network import NeuralNetwork
import numpy as np
import time

def load_trained_network(filepath='best_snake_brain.npz'):
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
    screen.title("Trained AI Snake")
    screen.tracer(0)
    
    snake = Snake()
    food = Food()
    
    score = 0
    steps = 0
    
    while steps < 2000:
        # Get state from the visual snake
        state = snake.get_state(food)
        
        # Network decides
        output = network.forward(state)
        action = np.argmax(output)
        
        # Apply action
        heading = snake.head.heading()
        if action == 1:  # Left
            new_heading = (heading + 90) % 360
            snake.head.setheading(new_heading)
        elif action == 2:  # Right
            new_heading = (heading - 90) % 360
            snake.head.setheading(new_heading)
        
        snake.move()
        screen.update()
        time.sleep(0.05)  # Adjust speed here
        steps += 1
        
        # Check food collision
        if snake.head.distance(food) < 15:
            food.refresh()
            snake.extend()
            score += 1
            print(f"Food eaten! Score: {score}")
        
        # Check wall collision
        if (snake.head.xcor() > 280 or snake.head.xcor() < -280 or 
            snake.head.ycor() > 280 or snake.head.ycor() < -280):
            print(f"Hit wall! Final score: {score}")
            break
        
        # Check tail collision
        for segment in snake.segments[1:]:
            if snake.head.distance(segment) < 10:
                print(f"Hit tail! Final score: {score}")
                break
    
    screen.exitonclick()

if __name__ == "__main__":
    print("Loading trained snake brain...")
    network = load_trained_network()
    print("Watching the AI play...")
    play_visual(network)