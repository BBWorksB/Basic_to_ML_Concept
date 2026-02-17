import random
import numpy as np

class GameEngine:
    """Pure Python snake game logic - no graphics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Start a new game."""
        # Snake starts at center, 3 segments
        self.snake_positions = [(0, 0), (-20, 0), (-40, 0)]
        self.heading = 0  # 0=right, 90=up, 180=left, 270=down
        
        # Food at random position
        self.food_pos = self._random_food_position()
        
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.game_over = False
    
    def _random_food_position(self):
        """Generate random food position."""
        x = random.randint(-13, 13) * 20  # -260 to 260 in steps of 20
        y = random.randint(-13, 13) * 20
        return (x, y)
    
    def get_state(self):
        """Return state vector [danger_straight, danger_left, danger_right, food_left, food_right, food_up, food_down]."""
        head_x, head_y = self.snake_positions[0]
        
        # Calculate positions one step ahead in each direction
        directions = {
            0:   ((20, 0),  (0, 20),  (0, -20)),   # RIGHT
            90:  ((0, 20),  (-20, 0), (20, 0)),    # UP
            180: ((-20, 0), (0, -20), (0, 20)),    # LEFT
            270: ((0, -20), (20, 0),  (-20, 0)),   # DOWN
        }
        
        straight_v, left_v, right_v = directions[self.heading]
        
        positions = {
            "straight": (head_x + straight_v[0], head_y + straight_v[1]),
            "left":     (head_x + left_v[0],     head_y + left_v[1]),
            "right":    (head_x + right_v[0],    head_y + right_v[1]),
        }
        
        # Check danger
        dangers = [self._is_danger(pos) for pos in positions.values()]
        
        # Check food direction
        food_x, food_y = self.food_pos
        food_direction = [
            1 if food_x < head_x else 0,  # food left
            1 if food_x > head_x else 0,  # food right
            1 if food_y > head_y else 0,  # food up
            1 if food_y < head_y else 0,  # food down
        ]
        
        return dangers + food_direction
    
    def _is_danger(self, pos):
        """Check if position is dangerous (wall or tail)."""
        x, y = pos
        
        # Wall collision
        if x > 280 or x < -280 or y > 280 or y < -280:
            return 1
        
        # Tail collision (check all body segments except head)
        for segment in self.snake_positions[1:]:
            if abs(x - segment[0]) < 10 and abs(y - segment[1]) < 10:
                return 1
        
        return 0
    
    def step(self, action):
        """Execute one game step. Action: 0=straight, 1=left, 2=right."""
        if self.game_over:
            return
        
        # Update heading based on action
        if action == 1:  # Turn left
            self.heading = (self.heading + 90) % 360
        elif action == 2:  # Turn right
            self.heading = (self.heading - 90) % 360
        
        # Calculate new head position
        head_x, head_y = self.snake_positions[0]
        if self.heading == 0:    # RIGHT
            new_head = (head_x + 20, head_y)
        elif self.heading == 90:  # UP
            new_head = (head_x, head_y + 20)
        elif self.heading == 180: # LEFT
            new_head = (head_x - 20, head_y)
        elif self.heading == 270: # DOWN
            new_head = (head_x, head_y - 20)
        
        # Move snake
        self.snake_positions.insert(0, new_head)
        
        # Check food collision
        if abs(new_head[0] - self.food_pos[0]) < 15 and abs(new_head[1] - self.food_pos[1]) < 15:
            self.score += 1
            self.steps_since_food = 0
            self.food_pos = self._random_food_position()
        else:
            # Remove tail if no food eaten
            self.snake_positions.pop()
        
        # Check collisions
        if self._is_danger(new_head):
            self.game_over = True
        
        self.steps += 1
        self.steps_since_food += 1
        
        # Timeout if wandering
        if self.steps_since_food > 200:
            self.game_over = True