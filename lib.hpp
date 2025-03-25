#pragma once

#include <stdint.h>

struct Vec2 {
    int32_t x;
    int32_t y;
};

struct Snake {
    Vec2 head;
    Vec2 body[3];

    Snake(int32_t* snake_pos) {
        head.x = snake_pos[0];
        head.y = snake_pos[1];
        for (int i = 0; i < 3; i++) {
            body[i].x = snake_pos[2 + i * 2];
            body[i].y = snake_pos[3 + i * 2];
        }
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

int32_t default_snake_move(Vec2 head, Vec2 body1);
int32_t snake_move_t1(int32_t* snake_pos, int32_t* food_pos);