#include <queue>
#include <stdint.h>
#include <map>
#include <vector>
#include <iostream>
#include "lib.hpp"

Vec2 operator+(const Vec2& lhs, const Vec2& rhs) {
    return {lhs.x + rhs.x, lhs.y + rhs.y};
}
Vec2 operator-(const Vec2& lhs, const Vec2& rhs) {
    return {lhs.x - rhs.x, lhs.y - rhs.y};
}
bool operator<(const Vec2& lhs, const Vec2& rhs) {
    return lhs.x == rhs.x ? lhs.y < rhs.y : lhs.x < rhs.x;
}
bool operator==(const Vec2& lhs, const Vec2& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y;
}
bool operator!=(const Vec2& lhs, const Vec2& rhs) {
    return !(lhs == rhs);
}

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

std::vector<Vec2> find_shortest_path(Snake& snake, Vec2 food, std::vector<Vec2>& barrier) {
    // Implement Dijkstra's algorithm or A* algorithm to find the shortest path
    std::vector<Vec2> path;

    std::queue<Snake> queue{};
    std::map<Vec2, Vec2> visited{};

    queue.push(snake);
    while (true) {
        if (queue.empty()) {
            break;
        }
        auto current_snake = queue.front();
        queue.pop();

        if (current_snake.head == food) {
            path.push_back(current_snake.head);
            break;
        }

        for (const auto& dir : {UP, DOWN, LEFT, RIGHT}) {
            Vec2 new_head = current_snake.head + dir2Vec(dir);
            if (new_head.x < 1 || new_head.x > 8 || new_head.y < 1 || new_head.y > 8) {
                continue; // Out of bounds
            }
            if (std::find(barrier.begin(), barrier.end(), new_head) != barrier.end()) {
                continue; // Hit a barrier
            }
            if (visited.find(new_head) != visited.end()) {
                continue; // Already visited
            }
            if (new_head.x == current_snake.body[0].x && new_head.y == current_snake.body[0].y) {
                continue; // Hit the snake's body
            }

            visited.insert({new_head, current_snake.head});
            Snake new_snake = Snake(std::vector<Vec2>{new_head, current_snake.head, current_snake.body[0], current_snake.body[1]});
            queue.push(new_snake);
        }
    }

    if (path.size() == 0) {
        return path; // No path found
    }

    while (path.front() != snake.head) {
        auto current = path.front();
        auto prev = visited[current];
        if (prev == snake.head) {
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

    return snake.to(path[0]);
}