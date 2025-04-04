import * as ort from 'onnxruntime-web';

let session = null;

export async function greedy_snake_step(board_size, snake, enemy_num, enemies, food_num, foods, round) {
    if (snake[0][0] === -1) {
        return "0x123456";
    }
    if (board_size === 5) {
        if (!session) {
            session = await ort.InferenceSession.create('./dqn_2p.onnx');
        }

        const input_tensor = create_state(board_size, snake, enemies, foods);
        console.log("Input Tensor:");
        for (let i = board_size - 1; i >= 0; i--) {
            let row = [];
            for (let j = 0; j < board_size; j++) {
                for (let k = 0; k < 8; k++) {
                    if (input_tensor[j * board_size * 8 + i * 8 + k] === 1) {
                        switch (k) {
                            case 0:
                                row.push(".");
                                break;
                            case 1:
                                row.push("H");
                                break;
                            case 2:
                                row.push("B");
                                break;
                            case 3:
                                row.push("T");
                                break;
                            case 4:
                                row.push("E");
                                break;
                            case 5:
                                row.push("e");
                                break;
                            case 6:
                                row.push("e");
                                break;
                            case 7:
                                row.push("F");
                                break;
                            default:
                                throw new Error("Invalid value");
                        }
                    }
                }
            }
            console.log(row);
        }
        const input = new ort.Tensor('float32', input_tensor, [1, board_size, board_size, 8]);
        const feeds = {};
        feeds[session.inputNames[0]] = input;
        const output = await session.run(feeds);
        const output_tensor = output[session.outputNames[0]].data;
        console.log("output_tensor", output_tensor);
        // find the max value in output_tensor
        let max_index = 0;
        let max_value = output_tensor[0];
        for (let i = 1; i < output_tensor.length; i++) {
            if (output_tensor[i] > max_value) {
                max_value = output_tensor[i];
                max_index = i;
            }
        }
        console.log("max_index", max_index);
        return max_index;
    }
}


function create_state(board_size, snake, enemies, foods) {
    const EMPTY = 0;
    const HEAD = 1;
    const BODY = 2;
    const TAIL = 3;
    const ENEMY_HEAD = 4;
    const ENEMY_BODY = 5;
    const ENEMY_TAIL = 6;
    const FOOD = 7;
    const tensor = new Float32Array(board_size * board_size * 8).fill(0);

    function set(x, y, value) {
        tensor[(x - 1) * board_size * 8 + (y - 1) * 8 + value] = 1;
    }

    set(snake[0], snake[1], HEAD);
    set(snake[2], snake[3], BODY);
    set(snake[4], snake[5], BODY);
    set(snake[6], snake[7], TAIL);
    
    if (enemies && enemies.length > 0) {
        for (let e = 0; e < enemies.length; e += 8) {
            if (e + 7 < enemies.length) {
                set(enemies[e], enemies[e + 1], ENEMY_HEAD);

                for (let i = 2; i < 6; i += 2) {
                    set(enemies[e + i], enemies[e + i + 1], ENEMY_BODY);
                }

                set(enemies[e + 6], enemies[e + 7], ENEMY_TAIL);
            }
        }
    }

    if (foods && foods.length > 0) {
        for (let f = 0; f < foods.length; f += 2) {
            if (f + 1 < foods.length) {
                set(foods[f], foods[f + 1], FOOD);
            }
        }
    }

    for (let x = 0; x < board_size; x++) {
        for (let y = 0; y < board_size; y++) {
            const
                hasFeature = tensor[x * board_size * 8 + y * 8 + HEAD] === 1 ||
                    tensor[x * board_size * 8 + y * 8 + BODY] === 1 ||
                    tensor[x * board_size * 8 + y * 8 + TAIL] === 1 ||
                    tensor[x * board_size * 8 + y * 8 + ENEMY_HEAD] === 1 ||
                    tensor[x * board_size * 8 + y * 8 + ENEMY_BODY] === 1 ||
                    tensor[x * board_size * 8 + y * 8 + ENEMY_TAIL] === 1 ||
                    tensor[x * board_size * 8 + y * 8 + FOOD] === 1;
            if (!hasFeature) {
                tensor[x * board_size * 8 + y * 8 + EMPTY] = 1;
            }
        }
    }

    return tensor;
}
