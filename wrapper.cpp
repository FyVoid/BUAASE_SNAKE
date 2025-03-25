#include "lib.hpp"

#include <emscripten/emscripten.h>

#ifdef __cplusplus
extern "C" {
#endif

EMSCRIPTEN_KEEPALIVE int32_t _func(int32_t* snake_pos, int32_t* food_pos) {
    return snake_move(snake_pos, food_pos);
}

#ifdef __cplusplus
}
#endif