#include <queue>
#include <stdint.h>
#include <unordered_map>
#include <vector>
#include <iostream>
#include "lib.hpp"

int32_t default_snake_move(Vec2 head, Vec2 body1) {
    if (head.x + 1 <= 8 && head.x + 1 != body1.x) {
        return RIGHT;
    } else if (head.x - 1 >= 1 && head.x + 1 != body1.x) {
        return LEFT;
    } else if (head.y + 1 >= 8 && head.y + 1 != body1.y) {
        return UP;
    } else if (head.y - 1 <= 1 && head.y - 1 != body1.y) {
        return DOWN;
    } else {
        return NONE;
    }
}

Vec2 dir2Vec(Direction dir) {
    switch (dir) {
        case UP: return {0, 1};
        case DOWN: return {0, -1};
        case LEFT: return {-1, 0};
        case RIGHT: return {1, 0};
        default: return {0, 0};
    }
}

template <typename T>
std::vector<T> make_vector(void* arr, int32_t size) {
    std::vector<T> vec(size);
    memcpy(vec.data(), arr, size * sizeof(T));
    return vec;
}

Tensor2D genBoard(const Snake& snake, const std::vector<Vec2>& foods, const std::vector<Vec2>& barriers) {
    Tensor2D board(1, 8 * 8 * 7);

    for (int32_t i = 0; i < 8; i++) {
        for (int32_t j = 0; j < 8; j++) {
            board.set(0, (i * 8 + j) * 7, 1);
        }
    }

    board.set(0, ((snake.head.x - 1) * 8 + (snake.head.y - 1)) * 7 + 1, 1);

    for (int32_t i = 0; i < 3; i++) {
        board.set(0, ((snake.body[i].x - 1) * 8 + (snake.body[i].y - 1)) * 7 + 2, 1);
    }

    for (int32_t i = 0; i < foods.size(); i++) {
        board.set(0, ((foods[i].x - 1) * 8 + (foods[i].y - 1)) * 7 + 6, 1);
    }

    for (int32_t i = 0; i < barriers.size(); i++) {
        board.set(0, ((barriers[i].x - 1) * 8 + (barriers[i].y - 1)) * 7 + 3, 1);
    }

    return board;
}

Direction tensor2Dir(const Tensor2D& out) {
    int32_t max_index = 0;
    double max_value = out.at(0, 0);
    for (int32_t i = 1; i < out.dim_y; i++) {
        if (out.at(0, i) > max_value) {
            max_value = out.at(0, i);
            max_index = i;
        }
    }

    switch (max_index) {
        case 0: return UP;
        case 1: return RIGHT;
        case 2: return DOWN;
        case 3: return LEFT;
        default: return NONE;
    }
}

int32_t snake_move_t1(int32_t* snake_pos, int32_t* food_pos) {
    auto snake = Snake(make_vector<Vec2>(snake_pos, 4));
    auto food = Vec2{food_pos[0], food_pos[1]};

    auto food_dir = food - snake.head;
    if (food_dir.x != 0) {
        auto move_pos = food_dir.x > 0 ? 1 : -1;
        if (snake.body[0].x != snake.head.x + move_pos) {
            return move_pos > 0 ? RIGHT : LEFT;
        }
    }
    if (food_dir.y != 0) {
        auto move_pos = food_dir.y > 0 ? 1 : -1;
        if (snake.body[0].y != snake.head.y + move_pos) {
            return move_pos > 0 ? UP : DOWN;
        }
    }

    return default_snake_move(snake.head, snake.body[0]);
}

std::vector<Snake> find_shortest_path(Snake& snake, Vec2 food, std::vector<Vec2>& barrier) {
    // Implement Dijkstra's algorithm or A* algorithm to find the shortest path
    std::vector<Snake> path;
    
    // // std::cout << "begin" << std::endl;

    std::priority_queue<std::tuple<Snake, int32_t>, std::vector<std::tuple<Snake, int32_t>>, DistComparer> queue{};
    std::unordered_map<Snake, Snake, std::hash<Snake>, SnakeValidComparer> visited{};

    queue.push({snake, 0});
    while (true) {
        if (queue.empty()) {
            break;
        }
        auto [current_snake, dist] = queue.top();
        queue.pop();

        if (current_snake.head == food) {
            path.push_back(current_snake);
            break;
        }

        // std::cout << "current_snake: " << current_snake.head.x << " " << current_snake.head.y << std::endl;
        // std::cout << "dist: " << dist << std::endl;

        for (const auto& dir : {UP, DOWN, LEFT, RIGHT}) {
            Vec2 new_head = current_snake.head + dir2Vec(dir);
            // // std::cout << "new_head: " << new_head.x << " " << new_head.y << std::endl;
            if (new_head.x < 1 || new_head.x > 8 || new_head.y < 1 || new_head.y > 8) {
                continue; // Out of bounds
            }
            if (std::find(barrier.begin(), barrier.end(), new_head) != barrier.end()) {
                continue; // Hit a barrier
            }
            if (new_head.x == current_snake.body[0].x && new_head.y == current_snake.body[0].y) {
                continue; // Hit the snake's body
            }
            Snake new_snake = Snake(std::vector<Vec2>{new_head, current_snake.head, current_snake.body[0], current_snake.body[1]});
            if (visited.find(new_snake) != visited.end()) {
                continue; // Already visited
            }

            // // std::cout << "take" << std::endl;
            visited.insert({new_snake, current_snake});
            queue.push({new_snake, dist + 1});
        }
    }

    if (path.size() == 0) {
        return path; // No path found
    }

    while (true) {
        auto current = path.front();
        auto prev = visited.find(current)->second;
        if (prev.head == snake.head && prev.body[0] == snake.body[0]) {
            break;
        }
        path.insert(path.begin(), prev);
    }

    return path;
}

int32_t snake_move_t2(int32_t* snake_pos, int32_t* food_pos, int32_t* barrier_pos) {
    auto snake = Snake(make_vector<Vec2>(snake_pos, 4));
    auto food = Vec2{food_pos[0], food_pos[1]};
    auto barrier = make_vector<Vec2>(barrier_pos, 12);

    // Apply djkstra
    auto path = find_shortest_path(snake, food, barrier);
    if (path.empty()) {
        return Direction::NONE;
    }

    return snake.to(path[0].head);
}