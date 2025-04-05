#include "lib.hpp"

#include <emscripten/emscripten.h>

#ifdef __cplusplus
extern "C" {
#endif

EMSCRIPTEN_KEEPALIVE int32_t _func_t1(int32_t* snake_pos, int32_t* food_pos) {
    return snake_move_t1(snake_pos, food_pos);
}

EMSCRIPTEN_KEEPALIVE int32_t _func_t2(int32_t* snake_pos, int32_t* food_pos, int32_t* barrier_pos) {
    return snake_move_t2(snake_pos, food_pos, barrier_pos);
}

EMSCRIPTEN_KEEPALIVE int32_t _func_t3(int32_t board_size, int32_t *snake_pos, int32_t enemy_count, int32_t *enemy_pos, int32_t food_num, int32_t *food_pos) {
    return snake_inference(board_size, snake_pos, enemy_count, enemy_pos, food_num, food_pos);
}

#ifdef __cplusplus
}
#endif