import torch
import torch.nn as nn

class DilatedInception(nn.Module):
    def __init__(self, input_dim, output_dim, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=3, dilation=dilation_factor, padding=dilation_factor)
        self.conv2 = nn.Conv1d(input_dim, output_dim, kernel_size=5, dilation=dilation_factor, padding=2*dilation_factor)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out = out1 + out2
        out = out.permute(0, 2, 1)
        return out
