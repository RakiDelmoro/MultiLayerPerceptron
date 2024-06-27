import torch
import statistics
from layer import linear_layer
from utils import print_correct_prediction, print_percentile_of_correct_probabilities, print_wrong_prediction

def mlp_network(feature_sizes: list, input_feature_size: int, device: str):
    layers = []
    parameters = []
    
    first_layer, first_layer_w, first_layer_b = linear_layer(input_feature_size, feature_sizes[0], device, "relu")
    layers.append(first_layer)
    parameters.extend([first_layer_w, first_layer_b])

    for i in range(len(feature_sizes)-1):
        input_feature = feature_sizes[i] 
        output_feature = feature_sizes[i+1]
        layer, w, b = linear_layer(input_feature, output_feature, device, "relu")
        layers.append(layer)
        parameters.extend([w, b])

    output_layer, output_layer_w, output_layer_b = linear_layer(feature_sizes[-1], 10, device="cuda", activation_function="softmax")
    parameters.extend([output_layer_w, output_layer_b])

    def forward(input_batch: torch.Tensor):
        previous_layer_output = input_batch
        for layer in layers:
            previous_layer_output = layer(previous_layer_output)

        return output_layer(previous_layer_output)
    
    def train_for_each_batch(dataloader, loss_function, optimizer):
        losses_for_each_batch = []
        for batch_image, batch_expected in dataloader:
            output_batch = forward(batch_image)
            network_loss = loss_function(output_batch, batch_expected)
            optimizer.zero_grad()
            network_loss.backward()
            optimizer.step()
            losses_for_each_batch.append(network_loss.item())

        return statistics.fmean(losses_for_each_batch)
        
    def check_model_outputs(model_outputs, expected):
        model_predictions = torch.concat(model_outputs)
        expected_model_predictions = torch.concat(expected)

        prediction_probabilities = []
        correct_predictions = []
        wrong_predictions = []
        model_accuracy = []
        for each in range(model_predictions.shape[0]):
            expected = expected_model_predictions[each]
            probability, predicted = model_predictions[each].max(dim=-1)
            correct_or_wrong = predicted.eq(expected).int().item()
            if predicted.item() == expected.item():
                predicted_and_expected = {'predicted': predicted.item(), 'expected': expected.item()}
                correct_predictions.append(predicted_and_expected)
            else:
                predicted_and_expected = {'predicted': predicted.item(), 'expected': expected.item()}
                wrong_predictions.append(predicted_and_expected)
            prediction_probabilities.append(probability.item())
            model_accuracy.append(correct_or_wrong)

        print_correct_prediction(correct_predictions, 5)
        print_wrong_prediction(wrong_predictions, 5)
        print_percentile_of_correct_probabilities(prediction_probabilities)

        correct_prediction_count = model_accuracy.count(1)
        wrong_prediction_count = model_accuracy.count(0)
        correct_percentage = (correct_prediction_count / len(model_predictions)) * 100
        wrong_percentage = (wrong_prediction_count / len(model_predictions)) * 100
        print(f"Correct percentage: {round(correct_percentage, 1)} Wrong percentage: {round(wrong_percentage, 1)}")

    def validate_for_each_batch(dataloader, loss_function):
        model_outputs = []
        expected = []
        losses_for_each_batch = []
        for batch_image, batch_expected in dataloader:
            output_batch = forward(batch_image)
            network_loss = loss_function(output_batch, batch_expected)
            losses_for_each_batch.append(network_loss)
            model_outputs.append(output_batch)
            expected.append(batch_expected)
        check_model_outputs(model_outputs, expected)
        return statistics.fmean(losses_for_each_batch)
    
    return (train_for_each_batch, validate_for_each_batch), parameters

def model_runner(network, training_loader, validation_loader, number_of_epochs, loss_function, optimizer):
    network_training_forward, network_validation_forward = network
    for epoch in range(number_of_epochs):
        average_loss_for_training_data = network_training_forward(training_loader, loss_function, optimizer)
        average_loss_for_validation_data = network_validation_forward(validation_loader, loss_function)
        print(f'EPOCH: {epoch+1} Training loss: {average_loss_for_training_data} Validation loss: {average_loss_for_validation_data}')