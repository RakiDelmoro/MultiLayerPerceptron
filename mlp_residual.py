import torch
from layer import linear_layer

def residual_mlp_network(feature_sizes: list, input_feature_size: int, device: str):
    layers = []
    parameters = []
    for i in range(len(feature_sizes)-1):
        input_feature = (feature_sizes[i] * 2) + 1
        output_feature = feature_sizes[i+1]
        layer, w, b = linear_layer(input_feature, output_feature, device, "relu")
        layers.append(layer)
        parameters.extend([[w, b]])

    first_layer, first_w, first_b = linear_layer((input_feature_size+239)+1, feature_sizes[0], device, "relu")
    layers.insert(0, first_layer)
    parameters.append([[first_w, first_b]])

    def pull_feature_from_layer_outputs(layer_outputs: list):
        idx_layer_output = [2 * 2**i for i in range(len(layer_outputs))]
        previous_layer_output = []
        for layer_output_idx, layer_output in enumerate(layer_outputs):
            if len(layer_outputs) < 2:
                previous_layer_output.append(layer_output)

            # TODO: Pulled layer output idx and divide 

        previous_layer_output_features = sum([each.shape[-1] for each in previous_layer_output])
        if previous_layer_output_features == 129:
            return torch.concat(previous_layer_output, dim=-1)
        else:
            fill_feature_size = sum([each.shape[-1] for each in previous_layer_output])
            previous_output = torch.zeros(layer_outputs[0].shape[0], 129-fill_feature_size, device="cuda")
            previous_layer_output.append(previous_output)
            return torch.concat(previous_layer_output, dim=-1)
    
    def forward(input_batch: torch.Tensor):
        previous_layer_output = torch.zeros((input_batch.shape[0], 239+1), device='cuda')
        layer_outputs = []
        for layer in layers:
            input_for_layer = torch.concat([input_batch, previous_layer_output], dim=-1)
            layer_output = layer(input_for_layer)
            layer_outputs.insert(0, layer_output)
            previous_layer_output = pull_feature_from_layer_outputs(layer_outputs)

    return forward

x = torch.randn(1, 784, device='cuda')
network = residual_mlp_network([128, 128, 128], 784, device="cuda")
print(network(x))