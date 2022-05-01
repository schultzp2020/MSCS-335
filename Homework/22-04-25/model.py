import torch
import torch.nn as nn

input_dim = 8
hidden_dim = [25000, 50, 25000]
output_dim = 7

def Net(input_dim, hidden_dims, output_dim, device):
    model =  nn.Sequential(
        nn.Linear(input_dim, hidden_dims[0]),
        nn.Tanh(),
        nn.BatchNorm1d(hidden_dims[0]),
        nn.Linear(hidden_dims[0], hidden_dims[1]),
        nn.LeakyReLU(),
        nn.BatchNorm1d(hidden_dims[1]),
        nn.Linear(hidden_dims[1], hidden_dims[2]),
        nn.ReLU(),
        nn.Linear(hidden_dims[2], output_dim),
    )
    model.to(device)
    return model