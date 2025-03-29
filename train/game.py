from queue import PriorityQueue
import torch
import random as rd

# Game one-hot vector representation
# empty [1, 0, 0, 0, 0, 0, 0]
# snake_head [0, 1, 0, 0, 0, 0, 0]
# snake_body [0, 0, 1, 0, 0, 0, 0]
# barrier [0, 0, 0, 1, 0, 0, 0]
# enemy_head [0, 0, 0, 0, 1, 0, 0]
# enemy_body [0, 0, 0, 0, 0, 1, 0]
# food [0, 0, 0, 0, 0, 0, 1]

def genBoard():
    board = torch.zeros(8, 8, 7)
    
    # Generate random snake position
    snake_x = rd.randint(0, 7)
    snake_y = rd.randint(0, 7)
    head = (snake_x, snake_y)
    board[snake_x][snake_y] = torch.tensor([0, 1, 0, 0, 0, 0, 0])
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    body = []
    for _ in range(3):
        valid_move = False
        while not valid_move:
            dx, dy = rd.choice(directions)
            new_x, new_y = snake_x + dx, snake_y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8 and torch.all(board[new_x][new_y] == 0):
                snake_x, snake_y = new_x, new_y
                body.append((snake_x, snake_y))
                board[snake_x][snake_y] = torch.tensor([0, 0, 1, 0, 0, 0, 0])
                valid_move = True
                
    # Generate random barriers
    barriers = []
    for _ in range(12):
        while True:
            barrier_x = rd.randint(0, 7)
            barrier_y = rd.randint(0, 7)
            if torch.all(board[barrier_x][barrier_y] == 0):
                board[barrier_x][barrier_y] = torch.tensor([0, 0, 0, 1, 0, 0, 0])
                barriers.append((barrier_x, barrier_y))
                break
            
    # Generate food
    foods = []
    for _ in range(5):
        while True:
            food_x = rd.randint(0, 7)
            food_y = rd.randint(0, 7)
            if torch.all(board[food_x][food_y] == 0):
                board[food_x][food_y] = torch.tensor([0, 0, 0, 0, 0, 0, 1])
                foods.append((food_x, food_y))
                break
            
    board = torch.where(torch.all(board == 0, dim=-1, keepdim=True), torch.tensor([1, 0, 0, 0, 0, 0, 0]), board)
    
    reward = torch.zeros(4)
    
    for i, choice in enumerate(directions):
        reward[i] = torch.tensor(calcReward(head, body, barriers, foods, choice))
        
    end = torch.where(reward < 0, torch.tensor(1), torch.tensor(0))
    
    next_board = torch.zeros(4, 8, 8, 7)
    for i, choice in enumerate(directions):
        next_head = (head[0] + choice[0], head[1] + choice[1])
        if 0 <= next_head[0] < 8 and 0 <= next_head[1] < 8 and torch.all(board[next_head[0]][next_head[1]] == 0):
            next_board[i] = board.clone()
            next_board[i][head[0]][head[1]] = torch.tensor([0, 0, 1, 0, 0, 0, 0])  # Current head becomes body
            next_board[i][next_head[0]][next_head[1]] = torch.tensor([0, 1, 0, 0, 0, 0, 0])  # New head position
            if body:
                tail = body[-1]  # Remove the last body segment
                next_board[i][tail[0]][tail[1]] = torch.tensor([1, 0, 0, 0, 0, 0, 0])  # Set it to empty
        else:
            next_board[i] = board.clone()  # Keep the board unchanged for invalid moves
        
    return board, reward, end, next_board

def calcDist2Food(head, barriers, food):
    # Initialize the priority queue
    pq = PriorityQueue()
    pq.put((0, head))  # (distance, position)
    visited = set()
    visited.add(head)
    
    # Directions for movement
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while not pq.empty():
        dist, current = pq.get()
        
        # If we reach the food, return the distance
        if current == food:
            return dist
        
        # Explore neighbors
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Check if the neighbor is within bounds and not a barrier or visited
            if 0 <= neighbor[0] < 8 and 0 <= neighbor[1] < 8 and neighbor not in visited:
                if neighbor not in barriers:
                    visited.add(neighbor)
                    pq.put((dist + 1, neighbor))
    
    # If no path to food is found, return a large value
    return float('inf')

def calcReward(head, body, barriers, foods, choice):
    # Check if snake move towards outside or into itself
    if head[0] + choice[0] > 8 or head[0] + choice[0] < 0 \
    or head[1] + choice[1] > 8 or head[1] + choice[1] < 0 \
    or (head[0] + choice[0] == body[0][0] and head[1] + choice[1] == body[0][1]):
        return -1
    
    # Check if snake move into barriers
    for barrier in barriers:
        if head[0] + choice[0] == barrier[0] and head[1] + choice[1] == barrier[1]:
            return -1
        
    oldDist = 0
    newDist = 0
    # Check if snake move into food
    for food in foods:
        if head[0] + choice[0] == food[0] and head[1] + choice[1] == food[1]:
            return 1
        
        oldDist += calcDist2Food(head, barriers, food)
        newDist += calcDist2Food((head[0] + choice[0], head[1] + choice[1]), barriers, food)

    if newDist < oldDist:
        return 0.8
    
    return 0.1
