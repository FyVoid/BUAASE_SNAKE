#pragma once

#include <vector>

struct Vec2 {
    int32_t x;
    int32_t y;
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

Vec2 operator+(const Vec2& lhs, const Vec2& rhs);
Vec2 operator-(const Vec2& lhs, const Vec2& rhs);
bool operator<(const Vec2& lhs, const Vec2& rhs);
bool operator==(const Vec2& lhs, const Vec2& rhs);
bool operator!=(const Vec2& lhs, const Vec2& rhs);

template <typename T>
std::vector<T> make_vector(void* arr, int32_t size);
Vec2 dir2Vec(Direction dir);

int32_t default_snake_move(Vec2 head, Vec2 body1);
int32_t snake_move_t1(int32_t* snake_pos, int32_t* food_pos);
std::vector<Vec2> find_shortest_path(Snake& snake, Vec2 food, std::vector<Vec2>& barrier);
int32_t snake_move_t2(int32_t* snake_pos, int32_t* food_pos, int32_t* barrier_pos);