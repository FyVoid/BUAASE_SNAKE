#include <stdint.h>
#include "lib.hpp"

Vec2 operator+(const Vec2& lhs, const Vec2& rhs) {
    return {lhs.x + rhs.x, lhs.y + rhs.y};
}
Vec2 operator-(const Vec2& lhs, const Vec2& rhs) {
    return {lhs.x - rhs.x, lhs.y - rhs.y};
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

int32_t snake_move_t2(int32_t* snake_pos, int32_t* food_pos, int32_t* barrier_pos) {
    auto snake = Snake(make_vector<Vec2>(snake_pos, 4));
    auto food = Vec2{food_pos[0], food_pos[1]};
    auto barrier = make_vector<Vec2>(barrier_pos, 12);

    // Apply djkstra
    
}