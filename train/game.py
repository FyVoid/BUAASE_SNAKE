from queue import Queue
from typing import Tuple, List, Optional, Set
import torch
import random as rd
from enum import Enum
import config

GRID_SIZE = config.GRID_SIZE
NUM_FOODS = config.NUM_FOODS
SNAKE_LENGTH = 4

# One-hot vector 
UNDEFINED_TENSOR = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32)
EMPTY_TENSOR = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32)
HEAD_TENSOR = torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float32)
BODY_TENSOR = torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float32)
ENEMY_HEAD_TENSOR = torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.float32)
ENEMY_BODY_TENSOR = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32)
FOOD_TENSOR = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32)


# Rewards
REWARD_FOOD = config.REWARD_FOOD
REWARD_DEATH = config.REWARD_DEATH
REWARD_CLOSER = config.REWARD_CLOSER
REWARD_STEP = config.REWARD_STEP
REWARD_KILL = config.REWARD_KILL

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
    
class EnemyFrame:
    def __init__(self, snake: List[Tuple[int, int]], alive: bool, state: torch.Tensor):
        self.snake = snake
        self.alive = alive
        self.state = state

class SnakeGame:
    def __init__(self, grid_size=GRID_SIZE, num_foods=NUM_FOODS, snake_length=SNAKE_LENGTH, enemy_snake_count=config.ENEMY_SNAKE_COUNT, game_mode=config.GAME_MODE):
        self.grid_size = grid_size
        self.num_foods = num_foods
        self.snake_length = max(1, snake_length) # Ensure at least length 1
        self.enemy_snake_count = enemy_snake_count
        self.game_mode = game_mode
        
        if self.game_mode == "1v1":
            self.init_positions = [
                [(0, 3), (0, 2), (0, 1), (0, 0)], # Snake 1
                [(4, 1), (4, 2), (4, 3), (4, 4)], # Snake 2
            ]
        else:
            self.init_positions = [
                [(3, 0), (2, 0), (1, 0), (0, 0)],
                [(7, 3), (7, 2), (7, 1), (7, 0)],
                [(4, 7), (5, 7), (6, 7), (7, 7)],
                [(0, 4), (0, 5), (0, 6), (0, 7)],
            ]

        self.board: torch.Tensor
        self.snake: List[Tuple[int, int]] = []
        self.enemies: List[EnemyFrame] = []
        self.foods: List[Tuple[int, int]] = []
        self.dead: bool = False
        self.total_steps: int = 0

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
            self.board = torch.zeros(self.grid_size, self.grid_size, config.STATE_FEATURES, dtype=torch.float32)
            self.enemies = [EnemyFrame([], True, torch.zeros(self.grid_size, self.grid_size, config.STATE_FEATURES, dtype=torch.float32)) for _ in range(self.enemy_snake_count)]
            self.snake = []
            self.foods = []
            self.total_steps = 0

            rd.shuffle(self.init_positions) # Shuffle the initial positions for randomness
            for i, position in enumerate(self.init_positions):
                if i == 0: # Player snake
                    for x, y in position:
                        self.snake.append((x, y))
                        self.board[x, y] = HEAD_TENSOR if len(self.snake) == 1 else BODY_TENSOR
                        for frame in self.enemies:
                            frame.state[x, y] = ENEMY_HEAD_TENSOR if len(self.snake) == 1 else ENEMY_BODY_TENSOR
                else: # Enemy snakes
                    for x, y in position:
                        self.enemies[i-1].snake.append((x, y))
                        self.board[x, y] = ENEMY_HEAD_TENSOR if len(self.enemies[i-1].snake) == 1 else ENEMY_BODY_TENSOR
                        for j, frame in enumerate(self.enemies):
                            if i - 1 == j:
                                frame.state[x, y] = HEAD_TENSOR if len(frame.snake) == 1 else BODY_TENSOR
                            else:
                                frame.state[x, y] = ENEMY_BODY_TENSOR if len(frame.snake) == 1 else ENEMY_BODY_TENSOR 

            while len(self.foods) < self.num_foods:
                x = rd.randint(0, self.grid_size - 1)
                y = rd.randint(0, self.grid_size - 1)
                if (x, y) not in self.snake and (x, y) not in self.foods and not any((x, y) in frame.snake for frame in self.enemies):
                    self.foods.append((x, y))
                    self.board[x, y] = FOOD_TENSOR
                    for frame in self.enemies:
                        frame.state[x, y] = FOOD_TENSOR

            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    cell = self.board[x, y]
                    if torch.equal(cell, UNDEFINED_TENSOR):
                        self.board[x, y] = EMPTY_TENSOR
                        for frame in self.enemies:
                            frame.state[x, y] = EMPTY_TENSOR
                        
            self.dead = False
            self.score = 0
            
            assert(len(self.snake) == self.snake_length)
            for frame in self.enemies:
                assert(len(frame.snake) == self.snake_length)
                
            self.drawBoards()

            return self.board.clone()

    def step(self, action_index: int, enemy_actions: List[int]) -> Tuple[torch.Tensor, float, bool]:
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
            return self.board.clone(), 0.0, True
        
        # TODO: rewrite this function
        
        reward = 0

        action = Direction.idx2dir(action_index)
        enemy_directions = [Direction.idx2dir(enemy_action) for enemy_action in enemy_actions]
        dx, dy = action.value[1]
        head = self.snake[0]
        new_head = (head[0] + dx, head[1] + dy)
        
        enemy_alive_flags = [frame.alive for frame in self.enemies]

        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size): # Boundary
            self.dead = True
            return self.board.clone(), REWARD_DEATH, True
        if new_head in self.snake[1:3]: # Self collision (ignore head itself)
            self.dead = True
            return self.board.clone(), REWARD_DEATH * 2, True
        for i, frame in enumerate(self.enemies):
            if enemy_alive_flags[i]:
                edx, edy = enemy_directions[i].value[1]
                enemy_head = frame.snake[0]
                enemy_new_head = (enemy_head[0] + edx, enemy_head[1] + edy)
                if enemy_new_head == new_head or new_head in frame.snake[0:3]: # Enemy collision (head-on or body)
                    self.dead = True
                    return self.board.clone(), REWARD_DEATH, True
                
                if not (0 <= enemy_new_head[0] < self.grid_size and 0 <= enemy_new_head[1] < self.grid_size):
                    frame.alive = False
                if enemy_new_head in frame.snake[1:3]:
                    frame.alive = False
                if enemy_new_head == new_head or enemy_new_head in self.snake[0:3]:
                    frame.alive = False
                    reward += REWARD_KILL
                for j, other_frame in enumerate(self.enemies):
                    if other_frame is not frame and enemy_alive_flags[j]:
                        odx, ody = enemy_directions[j].value[1]
                        other_enemy_head = other_frame.snake[0]
                        other_enemy_new_head = (other_enemy_head[0] + odx, other_enemy_head[1] + ody)
                        if enemy_new_head == other_enemy_new_head or enemy_new_head in other_frame.snake[1:3]:
                            frame.alive = False
                            break

        distances_before = [abs(head[0] - fx) + abs(head[1] - fy) for fx, fy in self.foods]
        distances_after = [abs(new_head[0] - fx) + abs(new_head[1] - fy) for fx, fy in self.foods]

        if distances_before and distances_after:
            sum_distance_before = min(distances_before)
            sum_distance_after = min(distances_after)
            if sum_distance_after < sum_distance_before:
                reward += sum_distance_after / sum_distance_before * REWARD_CLOSER

        if new_head in self.foods:
            self.foods.remove(new_head)
            reward += REWARD_FOOD
        else:
            reward += REWARD_STEP

        self.snake.insert(0, new_head)
        if len(self.snake) > 1: # Ensure there was an old head
            old_head = self.snake[1] # The segment that was previously the head
        tail = self.snake.pop()
            
        for i, frame in enumerate(self.enemies):
            if frame.alive:
            
                enemy_head = frame.snake[0]
                edx, edy = enemy_directions[i].value[1]
                enemy_new_head = (enemy_head[0] + edx, enemy_head[1] + edy)
                
                if enemy_new_head in self.foods:
                    self.foods.remove(enemy_new_head)
                
                frame.snake.insert(0, enemy_new_head)
                tail = frame.snake.pop()
        
        self.total_steps += 1
                            
        done = self.dead or self.total_steps >= config.MAX_STEP_PER_GAME

        while len(self.foods) < self.num_foods:
            x = rd.randint(0, self.grid_size - 1)
            y = rd.randint(0, self.grid_size - 1)
            if (x, y) in self.snake or (x, y) in self.foods:
                continue
            if any((x, y) in frame.snake and frame.alive for frame in self.enemies):
                continue
            self.foods.append((x, y))
            
        self.drawBoards() # Update the board with the new positions of the snakes and food

        return self.board.clone(), reward, done
    
    def drawBoards(self):
        # redraw all the boards
        self.board = torch.zeros(self.grid_size, self.grid_size, config.STATE_FEATURES, dtype=torch.float32)
        for frame in self.enemies:
            frame.state = torch.zeros(self.grid_size, self.grid_size, config.STATE_FEATURES, dtype=torch.float32)
        self.board[self.snake[0][0], self.snake[0][1]] = HEAD_TENSOR
        for frame in self.enemies:
            if frame.alive:
                frame.state[self.snake[0][0], self.snake[0][1]] = ENEMY_HEAD_TENSOR
        for i in range(1, len(self.snake)):
            self.board[self.snake[i][0], self.snake[i][1]] = BODY_TENSOR
            for frame in self.enemies:
                if frame.alive:
                    frame.state[self.snake[i][0], self.snake[i][1]] = ENEMY_BODY_TENSOR
        for food in self.foods:
            self.board[food[0], food[1]] = FOOD_TENSOR
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if torch.equal(self.board[x, y], UNDEFINED_TENSOR):
                    self.board[x, y] = EMPTY_TENSOR
        
        for frame in self.enemies:
            if frame.alive:
                frame.state[frame.snake[0][0], frame.snake[0][1]] = HEAD_TENSOR
                self.board[frame.snake[0][0], frame.snake[0][1]] = ENEMY_HEAD_TENSOR
                for i in range(1, len(frame.snake)):
                    frame.state[frame.snake[i][0], frame.snake[i][1]] = BODY_TENSOR
                    self.board[frame.snake[i][0], frame.snake[i][1]] = ENEMY_BODY_TENSOR
                for food in self.foods:
                    frame.state[food[0], food[1]] = FOOD_TENSOR
                for x in range(self.grid_size):
                    for y in range(self.grid_size):
                        if torch.equal(frame.state[x, y], UNDEFINED_TENSOR):
                            frame.state[x, y] = EMPTY_TENSOR
                for f in self.enemies:
                    if f is not frame and f.alive:
                        f.state[frame.snake[0][0], frame.snake[0][1]] = ENEMY_HEAD_TENSOR
                        for i in range(1, len(frame.snake)):
                            f.state[frame.snake[i][0], frame.snake[i][1]] = ENEMY_BODY_TENSOR
    
    def print(self):
        grid = [['.' for _ in range(self.grid_size * (self.enemy_snake_count + 2))] for _ in range(self.grid_size)]
        for i in range(self.grid_size):
            for k in range(1, self.enemy_snake_count + 1):
                grid[i][self.grid_size * k + k - 1] = '|'
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell = self.board[x, y]
                if torch.equal(cell, FOOD_TENSOR):
                    grid[self.grid_size - y - 1][x] = '*'
                elif torch.equal(cell, HEAD_TENSOR):
                    grid[self.grid_size - y - 1][x] = 'H'
                elif torch.equal(cell, BODY_TENSOR):
                    grid[self.grid_size - y - 1][x] = 'o'
                elif torch.equal(cell, ENEMY_HEAD_TENSOR):
                    grid[self.grid_size - y - 1][x] = 'E'
                elif torch.equal(cell, ENEMY_BODY_TENSOR):
                    grid[self.grid_size - y - 1][x] = 'e'
        
        for k in range(1, self.enemy_snake_count + 1):
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    cell = self.enemies[k - 1].state[x][y]
                    if torch.equal(cell, FOOD_TENSOR):
                        grid[self.grid_size - y - 1][x + self.grid_size * k + k] = '*'
                    elif torch.equal(cell, HEAD_TENSOR):
                        grid[self.grid_size - y - 1][x + self.grid_size * k + k] = 'H'
                    elif torch.equal(cell, BODY_TENSOR):
                        grid[self.grid_size - y - 1][x + self.grid_size * k + k] = 'o'
                    elif torch.equal(cell, ENEMY_HEAD_TENSOR):
                        grid[self.grid_size - y - 1][x + self.grid_size * k + k] = 'E'
                    elif torch.equal(cell, ENEMY_BODY_TENSOR):
                        grid[self.grid_size - y - 1][x + self.grid_size * k + k] = 'e'
                        
        for i in range(self.grid_size):
            print(''.join(grid[i]))
        print(f"Total Steps: {self.total_steps}")