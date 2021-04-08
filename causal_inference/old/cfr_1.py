import torch
import torch.nn as nn
import torch.nn.functional as F

class RepresentationNetwork(nn.Module):

    def __init__(self,
                 input_dim : int,
                 hidden_1_dim : int,
                 hidden_2_dim : int,
                 output_dim: int,
                 act_fn : object = nn.GELU):
        """
            - act_fn : Activation function used throughout the network
        """
        super().__init__()
        self.representation_net = nn.Sequential(
            nn.Linear(input_dim, hidden_1_dim), # 32x32 => 16x16
            act_fn(),
            nn.Linear(hidden_1_dim, hidden_2_dim),
            act_fn(),
            nn.Linear(hidden_2_dim, output_dim)
        )

    def forward(self, x):
        return self.representation_net(x)

class RegressionNetwork(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_1_dim: int,
                 hidden_2_dim: int,
                 output_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - act_fn : Activation function used throughout the network
        """
        super().__init__()
        self.treated = nn.Sequential(
            nn.Linear(input_dim, hidden_1_dim),
            act_fn(),
            nn.Linear(hidden_1_dim, hidden_2_dim),
            act_fn(),
            nn.Linear(hidden_2_dim, 1)
        )
        #self.control = nn.Sequential(
        #    nn.Linear(input_dim, hidden_1_dim),
        #    act_fn(),
        #    nn.Linear(hidden_1_dim, hidden_2_dim),
        #    act_fn(),
        #    nn.Linear(hidden_2_dim, 1)
        #)


    def forward(self, x):
        x = self.treated(x)
        #x = self.control(x)
        return x