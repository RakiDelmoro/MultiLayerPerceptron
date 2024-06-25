import torch
from torch.utils.data import DataLoader, TensorDataset
from mlp_residual import residual_mlp_network, model_runner
from dataloader import load_data_to_memory


def main():
    EPOCHS = 100
    DEVICE = "cuda"
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    BATCH_SIZE = 5000
    LEARNING_RATE = 0.01
    HIDDEN_LAYERS = [128] * 800
    IMAGE_FEATURE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
    MODEL, PARAMETERS = residual_mlp_network(feature_sizes=HIDDEN_LAYERS, input_feature_size=IMAGE_FEATURE_SIZE, device=DEVICE)
    LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
    OPTIMIZER = torch.optim.AdamW(PARAMETERS, lr=LEARNING_RATE)

    IMAGE_FOR_TRAIN, IMAGE_LABEL_FOR_TRAIN, IMAGE_FOR_VALIDATION, IMAGE_LABEL_FOR_VALIDATION = load_data_to_memory('./training-data/mnist.pkl.gz')
    TRAINING_DATALOADER = DataLoader(TensorDataset(IMAGE_FOR_TRAIN, IMAGE_LABEL_FOR_TRAIN), batch_size=BATCH_SIZE, shuffle=True)
    VALIDATION_DATALOADER = DataLoader(TensorDataset(IMAGE_FOR_VALIDATION, IMAGE_LABEL_FOR_VALIDATION), batch_size=250)

    model_runner(MODEL, TRAINING_DATALOADER, VALIDATION_DATALOADER, EPOCHS, LOSS_FUNCTION, OPTIMIZER)

main()
