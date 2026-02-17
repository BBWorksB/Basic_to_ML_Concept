from turtle import Screen
from snake import Snake
import time
from food import Food
from scoreboard import FONT, ScoreBoard

screen = Screen()
screen.setup(width=600, height=600)
screen.bgcolor("black")
screen.title("Snake Game")
screen.tracer(0)  # turns off the animation, so we can update the screen manually


food = Food()
snake = Snake()
scoreboard = ScoreBoard()


screen.listen()
screen.onkey(snake.up, "Up")
screen.onkey(snake.down, "Down")
screen.onkey(snake.left, "Left")
screen.onkey(snake.right, "Right")


game_is_on = True
while game_is_on:
    screen.update()
    time.sleep(0.1)

    snake.move()
    print(snake.get_state(food))
    

    # Detect collision with food
    if snake.head.distance(food) < 15:
        food.refresh()
        snake.extend()
        scoreboard.update_scoreboard()
        scoreboard.increase_score()

    # Detect collision with wall
    if snake.head.xcor() > 280 or snake.head.xcor() < -280 or snake.head.ycor() > 280 or snake.head.ycor() < -280:
        game_is_on = False
        scoreboard.game_over()

    # Detect with tails
    for segment in snake.segments[1:]:
        if snake.head.distance(segment) < 10:
            game_is_on = False
            scoreboard.game_over()

screen.exitonclick()   


# TODO control the snake
# def move_up():
#     segments[0].setheading(90)

# def move_down():
#     segments[0].setheading(270)

# def move_left():
#     segments[0].setheading(180)

# def move_right():
#     segments[0].setheading(0)

# screen.listen()
# screen.onkey(move_up, "Up")
# screen.onkey(move_down, "Down")
# screen.onkey(move_left, "Left")
# screen.onkey(move_right, "Right")

# TODO Detect collision with food



# TODO Create a scoreboard

# TODO Detect collision with wall and self/ Tail

# 