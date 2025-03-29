from queue import PriorityQueue
from typing import Tuple
import torch
import random as rd
from enum import Enum

# Game one-hot vector representation
# empty [1, 0, 0, 0, 0, 0, 0]
# snake_head [0, 1, 0, 0, 0, 0, 0]
# snake_body [0, 0, 1, 0, 0, 0, 0]
# barrier [0, 0, 0, 1, 0, 0, 0]
# enemy_head [0, 0, 0, 0, 1, 0, 0]
# enemy_body [0, 0, 0, 0, 0, 1, 0]
# food [0, 0, 0, 0, 0, 0, 1]

class Direction(Enum):
    U = (0, (0, 1))
    L = (1, (-1, 0))
    D = (2, (0, -1))
    R = (3, (1, 0))


class SnakeGame:
    def __init__(self):
        self.init_board = None
        self.init_snake = []
        self.init_foods = []
        self.init_barriers = []
        self.board = None
        self.snake = []
        self.foods = []
        self.barriers = []
        self.dead = False
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        self.genBoard()

    def genBoard(self):
        board = torch.zeros(8, 8, 7)
        snake_x = rd.randint(0, 7)
        snake_y = rd.randint(0, 7)
        head = (snake_x, snake_y)
        board[snake_x][snake_y] = torch.tensor([0, 1, 0, 0, 0, 0, 0])

        body = []
        for _ in range(3):
            valid_move = False
            while not valid_move:
                dx, dy = rd.choice(self.directions)
                new_x, new_y = snake_x + dx, snake_y + dy
                if 0 <= new_x < 8 and 0 <= new_y < 8 and torch.all(board[new_x][new_y] == 0):
                    snake_x, snake_y = new_x, new_y
                    body.append((snake_x, snake_y))
                    board[snake_x][snake_y] = torch.tensor([0, 0, 1, 0, 0, 0, 0])
                    valid_move = True
        self.init_snake = [head] + body
        self.snake = [head] + body

        barriers = []
        for _ in range(12):
            while True:
                barrier_x = rd.randint(0, 7)
                barrier_y = rd.randint(0, 7)
                if torch.all(board[barrier_x][barrier_y] == 0):
                    board[barrier_x][barrier_y] = torch.tensor([0, 0, 0, 1, 0, 0, 0])
                    barriers.append((barrier_x, barrier_y))
                    break
        self.init_barriers = barriers
        self.barriers = barriers


        foods = []
        for _ in range(5):
            while True:
                food_x = rd.randint(0, 7)
                food_y = rd.randint(0, 7)
                if torch.all(board[food_x][food_y] == 0):
                    board[food_x][food_y] = torch.tensor([0, 0, 0, 0, 0, 0, 1])
                    foods.append((food_x, food_y))
                    break
        self.init_foods = foods
        self.foods = foods

        
        board = torch.where(torch.all(board == 0, dim=-1, keepdim=True), torch.tensor([1, 0, 0, 0, 0, 0, 0]), board)

        self.init_board = board.clone()
        self.board = board
        self.dead = False


    def reset(self):
        self.board = self.init_board.clone()
        self.snake = self.init_snake
        self.foods = self.init_foods
        self.barriers = self.init_barriers
        self.dead = False

    def step(self, action: Direction) -> Tuple[torch.Tensor, float, bool]:
        if self.dead:
            return self.board, 0, True
        dx, dy = action.value[1]
        head = self.snake[0]
        new_head = (head[0] + dx, head[1] + dy)
        if not (0 <= new_head[0] < 8 and 0 <= new_head[1] < 8):
            self.dead = True
            return self.board, -1, True
        if new_head in self.snake[1:] or new_head in self.barriers:
            self.dead = True
            return self.board, -1, True
    
        next_board = torch.zeros(8, 8, 7)
        if 0 <= new_head[0] < 8 and 0 <= new_head[1] < 8 and torch.all(self.board[new_head[0]][new_head[1]] == 0):
            next_board = self.board.clone()
            next_board[head[0]][head[1]] = torch.tensor([0, 0, 1, 0, 0, 0, 0])  # Current head becomes body
            next_board[new_head[0]][new_head[1]] = torch.tensor([0, 1, 0, 0, 0, 0, 0])  # New head position
            self.snake.insert(0, new_head)  # Update snake position
            tail = self.snake.pop()  # Remove the tail
            next_board[tail[0]][tail[1]] = torch.tensor([1, 0, 0, 0, 0, 0, 0])  # Update tail position
        else:
            next_board = self.board.clone() 
        

        
        

if __name__ == "__main__":
    game = SnakeGame()
    print(game.board)
    game.reset()