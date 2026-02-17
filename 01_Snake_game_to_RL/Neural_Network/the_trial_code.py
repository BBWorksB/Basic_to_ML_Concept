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
         # 0=right, 90=up, 180=left, 270=down
        
        # YOUR JOB: calculate the position ONE step ahead in three directions:
        # - straight (current heading)
        # straight = ((head_x + MOVE_DISTANCE), head_y )     
        # - left (90 degrees left of current heading)  
        # left = (head_x, head_y + MOVE_DISTANCE)
        # - right (90 degrees right of current heading)
        # right = (head_x, head_y - MOVE_DISTANCE)
        
        # # Then for each position, check: is it a wall or a tail segment?
        # if heading == UP:
        #     straight = (head_x, head_y + MOVE_DISTANCE)
        #     left = (head_x - MOVE_DISTANCE, head_y)
        #     right = (head_x + MOVE_DISTANCE, head_y)
        # elif heading == DOWN:
        #     straight = (head_x, head_y - MOVE_DISTANCE)
        #     left = (head_x + MOVE_DISTANCE, head_y)
        #     right = (head_x - MOVE_DISTANCE, head_y)
        # elif heading == LEFT:
        #     straight = (head_x - MOVE_DISTANCE, head_y)
        #     left = (head_x, head_y - MOVE_DISTANCE)
        #     right = (head_x, head_y + MOVE_DISTANCE)
        # elif heading == RIGHT:
        #     straight = (head_x + MOVE_DISTANCE, head_y)
        #     left = (head_x, head_y + MOVE_DISTANCE)
        #     right = (head_x, head_y - MOVE_DISTANCE)
        # else:
        #     raise ValueError("Invalid heading value")

        # Direction vectors mapped by heading (straight, left, right)
        directions = {
            RIGHT: ((MOVE_DISTANCE, 0),  (0, MOVE_DISTANCE),  (0, -MOVE_DISTANCE)),
            UP:    ((0, MOVE_DISTANCE),  (-MOVE_DISTANCE, 0), (MOVE_DISTANCE, 0)),
            LEFT:  ((-MOVE_DISTANCE, 0), (0, -MOVE_DISTANCE), (0, MOVE_DISTANCE)),
            DOWN:  ((0, -MOVE_DISTANCE), (MOVE_DISTANCE, 0),  (-MOVE_DISTANCE, 0)),
        }

        straight_v, left_v, right_v = directions[heading]        

        # # Check for danger straight
        # if straight[0] > 280 or straight[0] < -280 or straight[1] > 280 or straight[1] < -280:
        #     danger_straight = 1
        # elif any(segment.distance(straight) < 10 for segment in self.segments[1:]):
        #     danger_straight = 1
        # else:
        #     danger_straight = 0

        # # Check for danger left
        # if left[0] > 280 or left[0] < -280 or left[1] > 280 or left[1] < -280:
        #     danger_left = 1
        # elif any(segment.distance(left) < 10 for segment in self.segments[1:]):
        #     danger_left = 1
        # else:
        #     danger_left = 0

        # # Check for danger right
        # if right[0] > 280 or right[0] < -280 or right[1] > 280 or right[1] < -280:
        #     danger_right = 1
        # elif any(segment.distance(right) < 10 for segment in self.segments[1:]):
        #     danger_right = 1
        # else:
        #     danger_right = 0

        # return [danger_straight, danger_left, danger_right]

        # Calculate predicted positions
        positions = {
            "straight": (head_x + straight_v[0], head_y + straight_v[1]),
            "left":     (head_x + left_v[0],     head_y + left_v[1]),
            "right":    (head_x + right_v[0],     head_y + right_v[1]),
        }


        # CHeck food location in relation to head
        food_direction = (
            1 if food.xcor() > head_x else 0,  # Food right of head
            1 if food.xcor() < head_x else 0,  # Food left of head
            1 if food.ycor() > head_y else 0   # Food above head
            1 if food.ycor() < head_y else 0   # Food below head
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