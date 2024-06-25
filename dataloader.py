import gzip
import torch
import pickle

def load_data_to_memory(file_name: str):
    with (gzip.open(file_name, 'rb')) as file:
        ((train_image_array, train_label_array), (validation_image_array, validation_label_array), _) = pickle.load(file, encoding='latin-1')

        return torch.tensor(train_image_array, device="cuda"), torch.tensor(train_label_array, device="cuda"), torch.tensor(validation_image_array, device="cuda"), torch.tensor(validation_label_array, device="cuda")