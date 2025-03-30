from queue import Queue
from typing import Tuple, List, Optional, Set
import torch
import random as rd
from enum import Enum

GRID_SIZE = 8
NUM_BARRIERS = 12
NUM_FOODS = 5
SNAKE_LENGTH = 4

# One-hot vector 
EMPTY_TENSOR = torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
HEAD_TENSOR = torch.tensor([0, 1, 0, 0, 0, 0, 0], dtype=torch.float32)
BODY_TENSOR = torch.tensor([0, 0, 1, 0, 0, 0, 0], dtype=torch.float32)
BARRIER_TENSOR = torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.float32)
# ENEMY_HEAD_TENSOR = torch.tensor([0, 0, 0, 0, 1, 0, 0], dtype=torch.float32) # Not used
# ENEMY_BODY_TENSOR = torch.tensor([0, 0, 0, 0, 0, 1, 0], dtype=torch.float32) # Not used
FOOD_TENSOR = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)

# Rewards
REWARD_FOOD = 10.0
REWARD_DEATH = -10.0
REWARD_STEP = -0.01

class Direction(Enum):
    # Value: (Action Index, (dx, dy)) - ensure indices match NN output (0, 1, 2, 3)
    U = (0, (0, 1))  # Assuming +y is Up
    L = (1, (-1, 0)) # Assuming -x is Left
    D = (2, (0, -1)) # Assuming -y is Down
    R = (3, (1, 0))  # Assuming +x is Right

    @staticmethod
    def idx2dir(index: int):
        for direction in Direction:
            if direction.value[0] == index:
                return direction
        raise ValueError(f"Invalid action index: {index}")

class SnakeGame:
    def __init__(self, grid_size=GRID_SIZE, num_barriers=NUM_BARRIERS, num_foods=NUM_FOODS, snake_length=SNAKE_LENGTH):
        self.grid_size = grid_size
        self.num_barriers = num_barriers
        self.num_foods = num_foods
        self.snake_length = max(1, snake_length) # Ensure at least length 1

        self.init_board: Optional[torch.Tensor] = None
        self.init_snake: List[Tuple[int, int]] = []
        self.init_foods: List[Tuple[int, int]] = []
        self.init_barriers: List[Tuple[int, int]] = []

        self.board: Optional[torch.Tensor] = None
        self.snake: List[Tuple[int, int]] = []
        self.foods: List[Tuple[int, int]] = []
        self.barriers: List[Tuple[int, int]] = []
        self.dead: bool = False
        self.score: int = 0

        # Possible moves (dx, dy) relative to current position
        self.possible_moves = [(0, 1), (-1, 0), (0, -1), (1, 0)] # Corresponds to U, L, D, R if mapped correctly

        self.genBoard() # Generate the initial board configuration

    def _is_reachable(self, start: Tuple[int, int], end: Tuple[int, int], obstacles: Set[Tuple[int, int]]) -> bool:
        """Checks reachability using BFS."""
        if start == end:
            return True
        q = Queue()
        q.put(start)
        visited = {start} | obstacles # Treat obstacles as visited initially

        while not q.empty():
            current = q.get()
            if current == end:
                return True

            for dx, dy in self.possible_moves:
                next_pos = (current[0] + dx, current[1] + dy)
                if (0 <= next_pos[0] < self.grid_size and
                        0 <= next_pos[1] < self.grid_size and
                        next_pos not in visited):
                    visited.add(next_pos)
                    q.put(next_pos)
        return False

    def genBoard(self):
        """Generates the initial board state including snake, barriers, and food."""
        while True: # Keep trying until a valid board with reachable food is generated
            board = torch.zeros(self.grid_size, self.grid_size, 7, dtype=torch.float32)
            snake = []
            barriers = []
            foods = []

            # 1. Place Snake Head
            head_x = rd.randint(0, self.grid_size - 1)
            head_y = rd.randint(0, self.grid_size - 1)
            head = (head_x, head_y)
            snake.append(head)
            board[head_x, head_y] = HEAD_TENSOR

            # 2. Place Snake Body
            current_pos = head
            placed_body = 0
            possible_directions = list(self.possible_moves)
            rd.shuffle(possible_directions)
            initial_direction = possible_directions[0]

            for _ in range(self.snake_length - 1):
                moved = False
                next_x, next_y = current_pos[0] + initial_direction[0], current_pos[1] + initial_direction[1]
                if (0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size and
                    (next_x, next_y) not in snake):
                     current_pos = (next_x, next_y)
                     snake.append(current_pos)
                     board[current_pos[0], current_pos[1]] = BODY_TENSOR
                     placed_body += 1
                     moved = True

                if not moved:
                    rd.shuffle(possible_directions)
                    for dx, dy in possible_directions:
                         next_x, next_y = current_pos[0] + dx, current_pos[1] + dy
                         if (0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size and
                             (next_x, next_y) not in snake):
                             current_pos = (next_x, next_y)
                             snake.append(current_pos)
                             board[current_pos[0], current_pos[1]] = BODY_TENSOR
                             placed_body += 1
                             moved = True
                             break
                if not moved:
                     break

            if placed_body < self.snake_length - 1:
                continue

            # 3. Place Barriers
            occupied = set(snake)
            placed_barriers = 0
            while placed_barriers < self.num_barriers:
                bx, by = rd.randint(0, self.grid_size - 1), rd.randint(0, self.grid_size - 1)
                pos = (bx, by)
                if pos not in occupied:
                    board[bx, by] = BARRIER_TENSOR
                    barriers.append(pos)
                    occupied.add(pos)
                    placed_barriers += 1

            # 4. Place Food
            placed_foods = 0

            potential_spots = []
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if (r, c) not in occupied:
                         potential_spots.append((r,c))
            rd.shuffle(potential_spots)

            for fx, fy in potential_spots:
                pos = (fx, fy)
                if self._is_reachable(head, pos, set(barriers) | set(snake[1:])):
                    board[fx, fy] = FOOD_TENSOR
                    foods.append(pos)
                    occupied.add(pos) 
                    placed_foods += 1
                    if placed_foods == self.num_foods:
                        break

            if placed_foods < self.num_foods:
                print(f"Warning: Could only place {placed_foods}/{self.num_foods} reachable foods. Board might be too crowded or generation failed.")
                if placed_foods == 0:
                    print("No reachable food spots found. Retrying board generation.")
                    continue

            # 5. Fill Empty Spaces
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if (r, c) not in occupied:
                        board[r, c] = EMPTY_TENSOR

            self.init_board = board.clone()
            self.init_snake = list(snake)
            self.init_foods = list(foods)
            self.init_barriers = list(barriers)
            self.reset()
            break

    def reset(self) -> torch.Tensor:
        if self.init_board is None:
            raise RuntimeError("Initial board not generated yet.")
        self.board = self.init_board.clone()
        self.snake = list(self.init_snake) # Use copies
        self.foods = list(self.init_foods)
        self.barriers = list(self.init_barriers)
        self.dead = False
        self.score = 0
        return self.board

    def step(self, action_index: int) -> Tuple[torch.Tensor, float, bool]:
        """
        Performs one step in the game based on the chosen action.

        Args:
            action_index (int): The index of the action to take (0: U, 1: L, 2: D, 3: R).

        Returns:
            Tuple[torch.Tensor, float, bool]: A tuple containing:
                - next_state (torch.Tensor): The board state after the action.
                - reward (float): The reward received for this step.
                - done (bool): Whether the game ended with this step.
        """
        if self.dead:
            # If already dead, return current state, 0 reward, and done=True
            return self.board, 0.0, True

        action = Direction.idx2dir(action_index)
        dx, dy = action.value[1]
        head = self.snake[0]
        new_head = (head[0] + dx, head[1] + dy)

        # 1. Check for collisions (Death conditions)
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size): # Boundary
            self.dead = True
            return self.board, REWARD_DEATH, True
        if new_head in self.barriers: # Barrier collision
             self.dead = True
             return self.board, REWARD_DEATH, True
        if new_head in self.snake[1:]: # Self collision (ignore head itself)
            self.dead = True
            return self.board, REWARD_DEATH, True

        # 2. Check for food
        if new_head in self.foods:
            self.foods.remove(new_head)
            reward = REWARD_FOOD
            self.score += 1
        else:
            reward = REWARD_STEP

        # 3. Update snake position
        self.snake.insert(0, new_head) # Add new head

        # 4. Update board tensor - New Head and Old Head becoming Body
        self.board[new_head[0], new_head[1]] = HEAD_TENSOR # Place new head
        if len(self.snake) > 1: # Ensure there was an old head
             old_head = self.snake[1] # The segment that was previously the head
             self.board[old_head[0], old_head[1]] = BODY_TENSOR # Old head becomes body

        tail = self.snake.pop()
        if self.board[tail[0], tail[1]][2] == 1: # Check if it's still marked as body
             self.board[tail[0], tail[1]] = EMPTY_TENSOR

        done = self.dead or len(self.foods) == 0 # Game over if dead or no food left

        return self.board, reward, done

    def printBoard(self):
        """Prints a simple text representation of the board."""
        if self.board is None:
            print("Board not initialized.")
            return

        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        # Draw barriers
        for bx, by in self.barriers:
             grid[by][bx] = '#' # Use # for barriers
        # Draw food
        for fx, fy in self.foods:
             grid[fy][fx] = '*' # Use * for food
        # Draw snake body
        for i, (sx, sy) in enumerate(self.snake):
             if i == 0:
                  grid[sy][sx] = 'H' # Use H for head
             else:
                  grid[sy][sx] = 'o' # Use o for body

        # Print grid (adjust y-axis inversion if needed based on your coordinate system)
        for y in range(self.grid_size - 1, -1, -1): # Print from top row (y=max) down
             row_str = ""
             for x in range(self.grid_size):
                  row_str += grid[y][x]
             print(row_str)
        print(f"Score: {self.score}, Dead: {self.dead}")


if __name__ == "__main__":
    game = SnakeGame() # Use fewer barriers
    print("Initial Board:")
    game.printBoard()

    # --- Example Usage ---
    import time
    done = False
    state = game.reset()
    total_reward = 0
    steps = 0
    while not done and steps < 100: # Limit steps for demo
        game.printBoard()
        # Choose a random valid action (replace with NN policy later)
        valid_actions = []
        current_head = game.snake[0]
        for action_idx in range(4):
            direction = Direction.idx2dir(action_idx)
            dx, dy = direction.value[1]
            next_head = (current_head[0] + dx, current_head[1] + dy)
            # Basic check to avoid immediate suicide (can be more sophisticated)
            if (0 <= next_head[0] < game.grid_size and
                0 <= next_head[1] < game.grid_size and
                next_head not in game.barriers and
                next_head not in game.snake[1:]):
                valid_actions.append(action_idx)

        if not valid_actions: # No safe moves, just pick one randomly
             action_index = rd.choice([0, 1, 2, 3])
             print(f"Step {steps+1}: No obvious safe moves, taking random action {Direction.idx2dir(action_index).name}")
        else:
             action_index = rd.choice(valid_actions)
             print(f"Step {steps+1}: Taking action {Direction.idx2dir(action_index).name}")


        next_state, reward, done = game.step(action_index)
        total_reward += reward
        steps += 1
        print(f"Reward: {reward:.2f}, Done: {done}, Total Reward: {total_reward:.2f}")
        time.sleep(0.5) # Pause for visualization

    print("\n--- Game Over ---")
    game.printBoard()
    print(f"Final Score: {game.score}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Steps: {steps}")