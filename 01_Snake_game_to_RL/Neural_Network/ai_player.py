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
        heading = snake.head.heading()

        if action == 0:  # Straight
            pass  # Don't change heading
        elif action == 1:  # Turn left (counterclockwise 90°)
            new_heading = (heading + 90) % 360
            snake.head.setheading(new_heading)
        elif action == 2:  # Turn right (clockwise 90°)
            new_heading = (heading - 90) % 360
            snake.head.setheading(new_heading)
    
        
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
        
        # Check collision with tail - FIXED
        hit_tail = False
        for segment in snake.segments[1:]:
            if snake.head.distance(segment) < 10:
                hit_tail = True
                break
        
        if hit_tail:
            break
        
        # Timeout if no food eaten recently (prevent infinite loops)
        if steps_since_food > 100:
            break
    
    # Fitness = score is priority, steps as tiebreaker
    return score * 1000 + steps
