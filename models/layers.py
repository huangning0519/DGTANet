import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedFusion, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        fused = torch.tanh(self.fc(x)) * torch.sigmoid(self.gate(x))
        return fused
