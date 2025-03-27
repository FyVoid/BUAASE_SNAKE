#pragma once

#include <vector>

struct Vec2 {
    int32_t x;
    int32_t y;
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
};

Vec2 operator+(const Vec2& lhs, const Vec2& rhs);
Vec2 operator-(const Vec2& lhs, const Vec2& rhs);

enum Direction {
    NONE = -1,
    UP = 0,
    LEFT = 1,
    DOWN = 2,
    RIGHT = 3
};

template <typename T>
std::vector<T> make_vector(void* arr, int32_t size);

int32_t default_snake_move(Vec2 head, Vec2 body1);
int32_t snake_move_t1(int32_t* snake_pos, int32_t* food_pos);
int32_t snake_move_t2(int32_t* snake_pos, int32_t* food_pos, int32_t* barrier_pos);