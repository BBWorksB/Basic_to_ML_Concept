'''
Docstring for snake:
the behaviour of the snake
'''
# from pyclbr import Class
from turtle import Turtle

STARTING_POSITIONS = [(0, 0), (-20, 0), (-40, 0)]
MOVE_DISTANCE = 20
UP = 90
DOWN = 270
LEFT = 180
RIGHT = 0

class Snake:

    def __init__(self):
        self.segments = []
        self.create_snake()
        self.head = self.segments[0]

    def create_snake(self):
        for position in STARTING_POSITIONS:
            self.add_segment(position)

    def add_segment(self, position):
            new_segment = Turtle("square")
            new_segment.color("white")
            new_segment.penup()
            new_segment.goto(position)
            self.segments.append(new_segment)

    def extend(self):
        # Add a new segment to the snake.
        self.add_segment(self.segments[-1].position())
            

    def move(self):
        for seg_num in range(len(self.segments) - 1, 0, -1):
            new_x = self.segments[seg_num - 1].xcor()
            new_y = self.segments[seg_num - 1].ycor()
            self.segments[seg_num].goto(new_x, new_y)

        self.head.forward(MOVE_DISTANCE)

    def up(self):
        if self.head.heading() != DOWN:
            self.head.setheading(UP)

    def down(self):
        if self.head.heading() != UP:
            self.head.setheading(DOWN)

    def left(self):
        if self.head.heading() != RIGHT:
            self.head.setheading(LEFT)

    def right(self):
        if self.head.heading() != LEFT:
            self.head.setheading(RIGHT)
    

    def get_state(self, food):
        """
        Returns the current state as a list of values the neural network will use.
        Start with just danger detection — 3 values.
        """
        # What is the head's current position?
        head_x = self.head.xcor()
        head_y = self.head.ycor()
        
        # What direction is the snake currently moving?
        heading = self.head.heading() 
        directions = {
            RIGHT: ((MOVE_DISTANCE, 0),  (0, MOVE_DISTANCE),  (0, -MOVE_DISTANCE)),
            UP:    ((0, MOVE_DISTANCE),  (-MOVE_DISTANCE, 0), (MOVE_DISTANCE, 0)),
            LEFT:  ((-MOVE_DISTANCE, 0), (0, -MOVE_DISTANCE), (0, MOVE_DISTANCE)),
            DOWN:  ((0, -MOVE_DISTANCE), (MOVE_DISTANCE, 0),  (-MOVE_DISTANCE, 0)),
        }

        straight_v, left_v, right_v = directions[heading]        

        # Calculate predicted positions
        positions = {
            "straight": (head_x + straight_v[0], head_y + straight_v[1]),
            "left":     (head_x + left_v[0],     head_y + left_v[1]),
            "right":    (head_x + right_v[0],     head_y + right_v[1]),
        }


        # CHeck food location in relation to head
        food_direction = (
        1 if food.xcor() < head_x else 0,  # Food left of head
        1 if food.xcor() > head_x else 0,  # Food right of head
        1 if food.ycor() > head_y else 0,  # Food above head
        1 if food.ycor() < head_y else 0,  # Food below head
        )

        return [self._is_danger(pos) for pos in positions.values()] + list(food_direction)
        

    def _is_danger(self, pos):
        """Check if a predicted position hits a wall or tail segment."""
        x, y = pos
        if x > 280 or x < -280 or y > 280 or y < -280:
            return 1
        if any(seg.distance(pos) < 10 for seg in self.segments[1:]):
            return 1
        return 0

