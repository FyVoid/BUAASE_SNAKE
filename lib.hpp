#pragma once

#include <functional>
#include <iostream>
#include <ostream>
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

#ifdef T3
class Tensor {
public:
    int32_t dim;
    std::vector<int32_t> shape;
    std::vector<float> data;

    Tensor(int32_t dim, std::vector<int32_t> shape) : dim(dim), shape(shape) {
        int32_t size = 1;
        for (int32_t s : shape) {
            size *= s;
        }
        data.resize(size);
    }

    ~Tensor() {
        // std::cout << "size:" << data.size() << std::endl;
        // std::cout << "tensor:";
        for (auto value : shape) {
            // std::cout << value << " ";
        }
        // std::cout << std::endl;
    }

    float& at(std::vector<int32_t> indices) {
        int32_t index = 0;
        int32_t stride = 1;
        for (int32_t i = dim - 1; i >= 0; --i) {
            index += indices[i] * stride;
            stride *= shape[i];
        }
        if (index < 0 || index >= static_cast<int32_t>(data.size())) {
            // std::cout << "shape:";
            for (auto value : shape) {
                // std::cout << value << " ";
            }
            // std::cout << std::endl;
            for (auto indice : indices) {
                // std::cout << indice << " ";
            }
            // std::cout << std::endl;
            throw std::out_of_range("Index out of bounds.");
        }
        return data[index];
    }
};

class Conv2DLayer {
public:
    Tensor weights;
    Tensor bias;
    int32_t kernel_size;
    int32_t stride;
    int32_t padding;
    int32_t in_channels;
    int32_t out_channels;

    Conv2DLayer(int32_t in_channels, int32_t out_channels, int32_t kernel_size, int32_t stride, int32_t padding)
        : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding),
          weights(4, {out_channels, in_channels, kernel_size, kernel_size}), bias(1, {out_channels}) {}

    Tensor forward(Tensor& input) {
        // std::cout << "Conv " << kernel_size << std::endl;
        int32_t input_height = input.shape[1];
        int32_t input_width = input.shape[2];
        int32_t output_height = (input_height - kernel_size + 2 * padding) / stride + 1;
        int32_t output_width = (input_width - kernel_size + 2 * padding) / stride + 1;

        Tensor output = Tensor(3, {out_channels, output_height, output_width});

        for (int32_t oc = 0; oc < out_channels; ++oc) {
            for (int32_t oh = 0; oh < output_height; ++oh) {
                for (int32_t ow = 0; ow < output_width; ++ow) {
                    float sum = bias.at({oc});
                    for (int32_t ic = 0; ic < in_channels; ++ic) {
                        for (int32_t kh = 0; kh < kernel_size; ++kh) {
                            for (int32_t kw = 0; kw < kernel_size; ++kw) {
                                int32_t ih = oh * stride + kh - padding;
                                int32_t iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                    sum += weights.at({oc, ic, kh, kw}) * input.at({ic, ih, iw});
                                }
                            }
                        }
                    }
                    output.at({oc, oh, ow}) = sum;
                }
            }
        }

        return output;
    }
};

class DenseLayer {
public:
    Tensor weights;
    Tensor bias;
    int32_t in_features;
    int32_t out_features;

    DenseLayer(int32_t in_features, int32_t out_features)
        : in_features(in_features), out_features(out_features),
          weights(2, {out_features, in_features}), bias(1, {out_features}) {}

    Tensor forward(Tensor& input) {
        // std::cout << "Dense " << std::endl;
        Tensor output(1, {out_features});
        for (int32_t i = 0; i < out_features; ++i) {
            output.at({i}) = bias.at({i});
            for (int32_t j = 0; j < in_features; ++j) {
                output.at({i}) += input.at({j}) * weights.at({i, j});
            }
        }

        return output;
    }
};

Tensor permute(Tensor& input, std::vector<int32_t> index);

Tensor relu(Tensor& input);

Tensor flatten(Tensor& input);

class Model {
public:
    Conv2DLayer conv1;
    Conv2DLayer conv2;
    Conv2DLayer conv3;
    DenseLayer dense1;

    Model(int32_t input_features)
        : conv1(8, 32, 3, 1, 1),
          conv2(32, 64, 5, 1, 2),
          conv3(64, 32, 3, 1, 1),
          dense1(32 * input_features * input_features, 4) {}

    Model
    (std::vector<float> conv1_weights,
          std::vector<float> conv1_bias,
          std::vector<float> conv2_weights,
          std::vector<float> conv2_bias,
          std::vector<float> conv3_weights,
          std::vector<float> conv3_bias,
          std::vector<float> dense1_weights,
          std::vector<float> dense1_bias
        )
        : conv1(8, 32, 3, 1, 1),
          conv2(32, 64, 5, 1, 2),
          conv3(64, 32, 3, 1, 1),
          dense1(32 * 8 * 8, 4) {
        // memcpy(conv1.weights.data.data(), conv1_weights.data(), conv1_weights.size() * sizeof(float));
        // memcpy(conv1.bias.data.data(), conv1_bias.data(), conv1_bias.size() * sizeof(float));
        // memcpy(conv2.weights.data.data(), conv2_weights.data(), conv2_weights.size() * sizeof(float));
        // memcpy(conv2.bias.data.data(), conv2_bias.data(), conv2_bias.size() * sizeof(float));
        // memcpy(conv3.weights.data.data(), conv3_weights.data(), conv3_weights.size() * sizeof(float));
        // memcpy(conv3.bias.data.data(), conv3_bias.data(), conv3_bias.size() * sizeof(float));
        // memcpy(dense1.weights.data.data(), dense1_weights.data(), dense1_weights.size() * sizeof(float));
        // memcpy(dense1.bias.data.data(), dense1_bias.data(), dense1_bias.size() * sizeof(float));
    }

    Tensor forward(Tensor& input) {
        for (auto value : input.shape) {
            // std::cout << value << " ";
        }
        // std::cout << std::endl;
        auto x = permute(input, {2, 0, 1});
        for (auto value : x.shape) {
            // std::cout << value << " ";
        }
        // std::cout << std::endl;
        x = conv1.forward(x);
        x = relu(x);
        x = conv2.forward(x);
        x = relu(x);
        x = conv3.forward(x);
        x = relu(x);
        x = permute(x, {1, 2, 0});
        auto result = flatten(x);
        for (auto value : result.shape) {
            // std::cout << value << " ";
        }
        auto ret = dense1.forward(result);
        return ret;
    }

};

void model_2p_init(Model& model);
    void model_4p_init(Model& model);
    int32_t snake_move_t3(int32_t board_size, int32_t* snake_pos, int32_t enemy_count, int32_t* enemy_pos, int32_t food_num, int32_t* food_pos);

#endif

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
#ifdef T1
int32_t snake_move_t1(int32_t* snake_pos, int32_t* food_pos);
#endif
#ifdef T2
std::vector<Snake> find_shortest_path(Snake& snake, Vec2 food, std::vector<Vec2>& barrier);
int32_t snake_move_t2(int32_t* snake_pos, int32_t* food_pos, int32_t* barrier_pos);
#endif
