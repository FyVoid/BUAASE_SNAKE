#pragma once

#include <functional>
#include <vector>

struct Vec2 {
    int32_t x;
    int32_t y;

    Vec2 operator+(const Vec2& rhs) const {
        return {this->x + rhs.x, this->y + rhs.y};
    }
    Vec2 operator-(const Vec2& rhs) const {
        return {this->x - rhs.x, this->y - rhs.y};
    }
    bool operator<(const Vec2& rhs) const {
        return this->x == rhs.x ? this->y < rhs.y : this->x < rhs.x;
    }
    bool operator==(const Vec2& rhs) const {
        return this->x == rhs.x && this->y == rhs.y;
    }
    bool operator!=(const Vec2& rhs) const {
        return !(*this == rhs);
    }
};

enum Direction {
    NONE = -1,
    UP = 0,
    LEFT = 1,
    DOWN = 2,
    RIGHT = 3
};

struct Snake {
    Vec2 head;
    Vec2 body[3];
    Snake(std::vector<Vec2> snake_pos) {
        head = snake_pos[0];
        body[0] = snake_pos[1];
        body[1] = snake_pos[2];
        body[2] = snake_pos[3];
    }
    Snake(const Snake& snake) {
        head = snake.head;
        body[0] = snake.body[0];
        body[1] = snake.body[1];
        body[2] = snake.body[2];
    }

    Direction to(Vec2 vec) {
        if (vec.x == head.x && vec.y == head.y + 1) {
            return UP;
        } else if (vec.x == head.x && vec.y == head.y - 1) {
            return DOWN;
        } else if (vec.x == head.x + 1 && vec.y == head.y) {
            return RIGHT;
        } else if (vec.x == head.x - 1 && vec.y == head.y) {
            return LEFT;
        }
        return NONE;
    }
};

namespace std {
    template <>
    struct hash<Vec2> {
        size_t operator()(const Vec2& vec) const {
            return std::hash<int32_t>()(vec.x) ^ std::hash<int32_t>()(vec.y);
        }
    };

    template <>
    struct hash<Snake> {
        size_t operator()(const Snake& snake) const {
            return std::hash<Vec2>()(snake.head) ^ std::hash<Vec2>()(snake.body[0]);
        }
    };
}

struct DistComparer {
    bool operator()(const std::tuple<Snake, int32_t>& lhs, const std::tuple<Snake, int32_t>& rhs) {
        return std::get<1>(lhs) > std::get<1>(rhs);
    }
};

struct SnakeValidComparer {
    bool operator()(const Snake& lhs, const Snake& rhs) const {
        return lhs.head == rhs.head && lhs.body[0] == rhs.body[0];
    }
};

template <typename T>
std::vector<T> make_vector(void* arr, int32_t size);
Vec2 dir2Vec(Direction dir);

int32_t default_snake_move(Vec2 head, Vec2 body1);
int32_t snake_move_t1(int32_t* snake_pos, int32_t* food_pos);
std::vector<Snake> find_shortest_path(Snake& snake, Vec2 food, std::vector<Vec2>& barrier);
int32_t snake_move_t2(int32_t* snake_pos, int32_t* food_pos, int32_t* barrier_pos);