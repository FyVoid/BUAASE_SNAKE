#include <cassert>
#include <cstring>
#include <queue>
#include <stdint.h>
#include <unordered_map>
#include <vector>
#include <iostream>
#include "lib.hpp"

#ifdef T3
#include "param.hpp"
#endif

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

#ifdef T3

Tensor permute(Tensor& input, std::vector<int32_t> index) {
    Tensor output = Tensor(input.dim, {input.shape[index[0]], input.shape[index[1]], input.shape[index[2]]});

    int32_t num_dims = input.shape.size();
    std::vector<int32_t> new_shape(num_dims);
    std::vector<int32_t> strides(num_dims, 1);

    for (int32_t i = 0; i < output.shape[0]; i++) {
        for (int32_t j = 0; j < output.shape[1]; j++) {
            for (int32_t k = 0; k < output.shape[2]; k++) {
                std::vector<int32_t> old_index(num_dims);
                old_index[index[0]] = i;
                old_index[index[1]] = j;
                old_index[index[2]] = k;
                output.at({i, j, k}) = input.at(old_index);
            }
        }
    }

    return output;
}

Tensor relu(Tensor& input) {
    auto shape = input.shape;

    Tensor output = Tensor(input.dim, {shape[0], shape[1], shape[2]});

    for (int32_t i = 0; i < input.data.size(); ++i) {
        output.data[i] = std::max(0.0f, input.data[i]);
    }

    return output;
}

Tensor flatten(Tensor& input) {
    auto shape = input.shape;

    Tensor output = Tensor(1, {shape[0] * shape[1] * shape[2]});

    for (int32_t i = 0; i < shape[0]; ++i) {
        for (int32_t j = 0; j < shape[1]; ++j) {
            for (int32_t k = 0; k < shape[2]; ++k) {
                output.data[i * shape[1] * shape[2] + j * shape[2] + k] = input.data[i * shape[1] * shape[2] + j * shape[2] + k];
            }
        }
    }
    
    return output;
}

enum GridType {
    EMPTY = 0,
    HEAD,
    BODY,
    TAIL,
    EHEAD,
    EBODY,
    ETAIL,
    FOOD,
};

void model_2p_init(Model& model) {
    assert(model.conv1.weights.data.size() == CONV1_WEIGHT_2P.size());
    assert(model.conv1.bias.data.size() == CONV1_BIAS_2P.size());
    assert(model.conv2.weights.data.size() == CONV2_WEIGHT_2P.size());
    assert(model.conv2.bias.data.size() == CONV2_BIAS_2P.size());
    assert(model.conv3.weights.data.size() == CONV3_WEIGHT_2P.size());
    assert(model.conv3.bias.data.size() == CONV3_BIAS_2P.size());
    // std::cout << model.dense1.weights.data.size() << " " << DENSE_WEIGHT_2P.size() << std::endl;
    assert(model.dense1.weights.data.size() == DENSE_WEIGHT_2P.size());
    assert(model.dense1.bias.data.size() == DENSE_BIAS_2P.size());
    memcpy(model.conv1.weights.data.data(), CONV1_WEIGHT_2P.data(), CONV1_WEIGHT_2P.size() * sizeof(float));
    memcpy(model.conv1.bias.data.data(), CONV1_BIAS_2P.data(), CONV1_BIAS_2P.size() * sizeof(float));
    memcpy(model.conv2.weights.data.data(), CONV2_WEIGHT_2P.data(), CONV2_WEIGHT_2P.size() * sizeof(float));
    memcpy(model.conv2.bias.data.data(), CONV2_BIAS_2P.data(), CONV2_BIAS_2P.size() * sizeof(float));
    memcpy(model.conv3.weights.data.data(), CONV3_WEIGHT_2P.data(), CONV3_WEIGHT_2P.size() * sizeof(float));
    memcpy(model.conv3.bias.data.data(), CONV3_BIAS_2P.data(), CONV3_BIAS_2P.size() * sizeof(float));
    memcpy(model.dense1.weights.data.data(), DENSE_WEIGHT_2P.data(), DENSE_WEIGHT_2P.size() * sizeof(float));
    memcpy(model.dense1.bias.data.data(), DENSE_BIAS_2P.data(), DENSE_BIAS_2P.size() * sizeof(float));
}

void model_4p_init(Model& model) {
    assert(model.conv1.weights.data.size() == CONV1_WEIGHT_4P.size());
    assert(model.conv1.bias.data.size() == CONV1_BIAS_4P.size());
    assert(model.conv2.weights.data.size() == CONV2_WEIGHT_4P.size());
    assert(model.conv2.bias.data.size() == CONV2_BIAS_4P.size());
    assert(model.conv3.weights.data.size() == CONV3_WEIGHT_4P.size());
    assert(model.conv3.bias.data.size() == CONV3_BIAS_4P.size());
    // std::cout << model.dense1.weights.data.size() << " " << DENSE_WEIGHT_2P.size() << std::endl;
    assert(model.dense1.weights.data.size() == DENSE_WEIGHT_4P.size());
    assert(model.dense1.bias.data.size() == DENSE_BIAS_4P.size());
    memcpy(model.conv1.weights.data.data(), CONV1_WEIGHT_4P.data(), CONV1_WEIGHT_4P.size() * sizeof(float));
    memcpy(model.conv1.bias.data.data(), CONV1_BIAS_4P.data(), CONV1_BIAS_4P.size() * sizeof(float));
    memcpy(model.conv2.weights.data.data(), CONV2_WEIGHT_4P.data(), CONV2_WEIGHT_4P.size() * sizeof(float));
    memcpy(model.conv2.bias.data.data(), CONV2_BIAS_4P.data(), CONV2_BIAS_4P.size() * sizeof(float));
    memcpy(model.conv3.weights.data.data(), CONV3_WEIGHT_4P.data(), CONV3_WEIGHT_4P.size() * sizeof(float));
    memcpy(model.conv3.bias.data.data(), CONV3_BIAS_4P.data(), CONV3_BIAS_4P.size() * sizeof(float));
    memcpy(model.dense1.weights.data.data(), DENSE_WEIGHT_4P.data(), DENSE_WEIGHT_4P.size() * sizeof(float));
    memcpy(model.dense1.bias.data.data(), DENSE_BIAS_4P.data(), DENSE_BIAS_4P.size() * sizeof(float));
}

int32_t snake_move_t3(int32_t board_size, int32_t* snake_pos, int32_t enemy_count, int32_t* enemy_pos, int32_t food_num, int32_t* food_pos) {
    Tensor input = Tensor(3, {board_size, board_size, 8});
    Model model(board_size);
    if (board_size == 5) {
        model_2p_init(model);
    } else if (board_size == 8) {
        model_4p_init(model);
    }
    auto set = [&](int32_t x, int32_t y, int32_t type) {
        input.at({x-1, y-1, type}) = 1.0f;
    };
    auto pair_apply = [&](int32_t* base, int32_t offset, auto fn, int32_t data) {
        fn(base[0 + offset * 2], base[1 + offset * 2], data);
    };
    pair_apply(snake_pos, 0, set, HEAD);
    pair_apply(snake_pos, 1, set, BODY);
    pair_apply(snake_pos, 2, set, BODY);
    pair_apply(snake_pos, 3, set, TAIL);

    for (int32_t i = 0; i < enemy_count; ++i) {
        pair_apply(enemy_pos, i * 4, set, EHEAD);
        pair_apply(enemy_pos, i * 4 + 1, set, EBODY);
        pair_apply(enemy_pos, i * 4 + 2, set, EBODY);
        pair_apply(enemy_pos, i * 4 + 3, set, ETAIL);
    }

    for (int32_t i = 0; i < food_num; ++i) {
        pair_apply(food_pos, i, set, FOOD);
    }

    for (int32_t i = 0; i < board_size; ++i) {
        for (int32_t j = 0; j < board_size; ++j) {
            bool flag = false;
            for (int32_t k = 0; k < 8; ++k) {
                if (input.at({i, j, k}) > 0.0f) {
                    flag = true;
                    break;
                }
            }
            if (!flag) {
                input.at({i, j, EMPTY}) = 1.0f;
            }
        }
    }

    auto result = model.forward(input);
    float max_value = -1e9;
    int32_t max_index = -1;
    for (auto i = 0; i < 4; ++i) {
        // std::cout << result.at({i}) << " ";
        if (result.at({i}) > max_value) {
            max_value = result.at({i});
            max_index = i;
        }
    }
    // std::cout << "mamba out" << std::endl;
    // std::cout << max_index << std::endl;
    return max_index;
    // return 0;
}

#endif

#ifdef T1

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

#endif

#ifdef T2

std::vector<Snake> find_shortest_path(Snake& snake, Vec2 food, std::vector<Vec2>& barrier) {
    // Implement Dijkstra's algorithm or A* algorithm to find the shortest path
    std::vector<Snake> path;
    
    // // // std::cout << "begin" << std::endl;

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

        // // std::cout << "current_snake: " << current_snake.head.x << " " << current_snake.head.y << std::endl;
        // // std::cout << "dist: " << dist << std::endl;

        for (const auto& dir : {UP, DOWN, LEFT, RIGHT}) {
            Vec2 new_head = current_snake.head + dir2Vec(dir);
            // // // std::cout << "new_head: " << new_head.x << " " << new_head.y << std::endl;
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

            // // // std::cout << "take" << std::endl;
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

#endif