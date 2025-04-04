#include <stdint.h>
#include "lib.hpp"

Tensor2D operator*(const Tensor2D& lhs, const Tensor2D& rhs) {
    Tensor2D result(lhs.dim_x, rhs.dim_y);
    for (int32_t i = 0; i < lhs.dim_x; i++) {
        for (int32_t j = 0; j < rhs.dim_y; j++) {
            for (int32_t k = 0; k < lhs.dim_y; k++) {
                result.set(i, j, lhs.at(i, k) * rhs.at(k, j));
            }
        }
    }
    
    return result;
}

Tensor2D operator+(const Tensor2D& lhs, const Tensor2D& rhs) {
    Tensor2D result(lhs.dim_x, rhs.dim_y);
    for (int32_t i = 0; i < lhs.dim_x; i++) {
        for (int32_t j = 0; j < rhs.dim_y; j++) {
            result.set(i, j, lhs.at(i, j) + rhs.at(i, j));
        }
    }
    
    return result;
}

Tensor2D ReLU(const Tensor2D& tensor) {
    Tensor2D result(tensor.dim_x, tensor.dim_y);
    for (int32_t i = 0; i < tensor.dim_x; i++) {
        for (int32_t j = 0; j < tensor.dim_y; j++) {
            result.set(i, j, std::max(0.0, tensor.at(i, j)));
        }
    }
    
    return result;
}

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

int32_t snake_move_t2(int32_t* snake_pos, int32_t* food_pos, int32_t* barrier_pos) {
    auto snake = Snake(make_vector<Vec2>(snake_pos, 4));
    auto food = Vec2{food_pos[0], food_pos[1]};
    auto barrier = make_vector<Vec2>(barrier_pos, 12);

    auto board = genBoard(snake, {food}, barrier);
    auto choice = tensor2Dir(Model::getInstance().forward(board));

    return choice;
}