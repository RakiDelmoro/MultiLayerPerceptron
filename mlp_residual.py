import torch
from layer import linear_layer

def residual_mlp_network(feature_sizes: list, input_feature_size: int, device: str):
    layers = []
    parameters = []
    
    first_layer, first_layer_w, first_layer_b = linear_layer((input_feature_size+239)+1, feature_sizes[0], device, "relu")
    layers.append(first_layer)
    parameters.extend([first_layer_w, first_layer_b])

    for i in range(len(feature_sizes)-1):
        input_feature = (feature_sizes[i] * 2) + 1
        output_feature = feature_sizes[i+1]
        layer, w, b = linear_layer(input_feature, output_feature, device, "relu")
        layers.append(layer)
        parameters.extend([w, b])

    output_layer, output_layer_w, output_layer_b = linear_layer(feature_sizes[-1]*2+1, 10, device="cuda", activation_function="softmax")
    parameters.extend([output_layer_w, output_layer_b])

    def pull_feature_from_layer_outputs(layer_outputs: list):
        # [0, 2, 4, 8, 16, 32, 64, 128]
        layer_output_idx_to_be_pulled = [0 if i == 0 else 2 * 2**(i-1) for i in range(8)]
        previous_layer_output = []
        for layer_output_idx, layer_output in enumerate(layer_outputs):
            is_layer_output_to_be_pulled = layer_output_idx in layer_output_idx_to_be_pulled
            if is_layer_output_to_be_pulled:
                if layer_output_idx == 0:
                    previous_layer_output.append(layer_output)
                else:
                    layer_feature_pulled = layer_output.shape[-1] // layer_output_idx
                    previous_layer_output.append(layer_output[:, :layer_feature_pulled])

        pulled_layer_output_total_features = sum([each.shape[-1] for each in previous_layer_output])
        if pulled_layer_output_total_features == feature_sizes[0]*2+1:
            return torch.concat(previous_layer_output, dim=-1)
        else:
            previous_output = torch.zeros(layer_outputs[0].shape[0], feature_sizes[0]*2+1-pulled_layer_output_total_features, device="cuda")
            previous_layer_output.append(previous_output)
            return torch.concat(previous_layer_output, dim=-1)

    def forward(input_batch: torch.Tensor):
        previous_layer_output = torch.zeros((input_batch.shape[0], 1024-input_feature_size), device='cuda')
        input_for_layer = torch.concat([input_batch, previous_layer_output], dim=-1)
        layer_outputs = []
        for layer in layers:
            layer_output = layer(input_for_layer)
            layer_outputs.insert(0, layer_output)
            input_for_layer = pull_feature_from_layer_outputs(layer_outputs)

        return output_layer(input_for_layer)

    return forward

x = torch.randn(1, 784, device='cuda')
hidden_layers = [128] * 800
network = residual_mlp_network(hidden_layers, 784, device="cuda")
print(network(x))
