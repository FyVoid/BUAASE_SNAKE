// 从编译生成的 cjs 文件中导入 Wasm Module
import Module from './module_t3.cjs'

// 考虑到真实网页场景，通过网络加载 wasm 可能很慢，所以 Module 是一个异步函数，
// 而我们是本地的环境，这里直接 await 就好
const wasm = await Module();
// 使用 cwrap 函数方便的包装 C 中的函数
// 使用方法：cwrap(函数名, 返回值类型, 参数列表)
const c_func = wasm.cwrap('_func_t3', 'number', ['number', 'array', 'number', 'array', 'number', 'array']);

// 测试时真正调用的方法
export const greedy_snake_move = (board_size, snake_pos, enemy_num, enemy_pos, food_num, food_pos) => {
    // 由于 seq 这样的 js数组 没有对应的C语言类型，
    // 而C语言的数组入参均表现为指针，所以需要包装一下
    let snake_array = new Uint8Array((new Int32Array(snake_pos)).buffer)
    let enemy_array = new Uint8Array((new Int32Array(enemy_pos)).buffer)
    let food_array = new Uint8Array((new Int32Array(food_pos)).buffer)
    return c_func(board_size, snake_array, enemy_num, enemy_array, food_num, food_array);
};