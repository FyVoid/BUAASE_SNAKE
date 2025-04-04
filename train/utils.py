# utils.py
import torch
import numpy as np
import config

def save_simple_params(model: torch.nn.Module, filename=config.PARAMS_SAVE_PATH):
    """
    Saves model parameters (weights and biases) in a simple binary format.
    Format: [num_layers,
             layer1_weights.shape[0], layer1_weights.shape[1], layer1_weights_flat..., layer1_bias...,
             layer2_weights.shape[0], layer2_weights.shape[1], layer2_weights_flat..., layer2_bias...,
             ...]
    """
    params = []
    shapes = []

    # Only save Linear layers for simplicity, assuming MLP structure from model.py
    linear_layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]

    num_layers = len(linear_layers)
    params.append(np.array([num_layers], dtype=np.int32)) # Store number of layers first

    for layer in linear_layers:
        weights = layer.weight.data.cpu().numpy()
        biases = layer.bias.data.cpu().numpy()

        # Store shapes
        shapes.append(np.array(weights.shape, dtype=np.int32)) # (out_features, in_features)
        shapes.append(np.array(biases.shape, dtype=np.int32)) # (out_features,)

        # Store flattened data
        params.append(weights.flatten().astype(np.float32))
        params.append(biases.flatten().astype(np.float32))

    # Combine shapes and data for saving
    all_data_to_save = params[0:1] # Start with num_layers
    param_idx = 1
    for weight_shape, bias_shape in zip(shapes[::2], shapes[1::2]):
         all_data_to_save.append(weight_shape)       # Add weight shape
         all_data_to_save.append(params[param_idx])  # Add flattened weights
         param_idx += 1
         all_data_to_save.append(bias_shape)         # Add bias shape
         all_data_to_save.append(params[param_idx])  # Add flattened biases
         param_idx += 1


    # Concatenate all numpy arrays
    final_array = np.concatenate([arr.reshape(-1) for arr in all_data_to_save])

    # Save to binary file
    with open(filename, 'wb') as f:
        f.write(final_array.tobytes())

    print(f"Parameters saved to {filename}")
    print(f"Format Hint: num_layers(int32), [w_shape(2*int32), w_data(float32...), b_shape(1*int32), b_data(float32...)] repeating")

# Example of how to potentially load (implementation in non-Python needed)
# def load_simple_params_example(filename=config.PARAMS_SAVE_PATH):
#     with open(filename, 'rb') as f:
#         raw_data = f.read()
#
#     # Example interpretation (pseudo-code for C/C++):
#     # ptr = raw_data
#     # num_layers = read_int32(ptr); ptr += 4
#     # for i in range(num_layers):
#     #     w_rows = read_int32(ptr); ptr += 4
#     #     w_cols = read_int32(ptr); ptr += 4
#     #     num_w_elements = w_rows * w_cols
#     #     weights = read_float32_array(ptr, num_w_elements); ptr += num_w_elements * 4
#     #     reshape weights to (w_rows, w_cols)
#     #
#     #     b_elements = read_int32(ptr); ptr += 4
#     #     biases = read_float32_array(ptr, b_elements); ptr += b_elements * 4
#     #     reshape biases to (b_elements,)
#     #     store/use weights and biases
#     pass