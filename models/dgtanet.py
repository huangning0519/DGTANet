import torch
import torch.nn as nn
from models.graph_learner import GraphLearner
from models.dilated_conv import DilatedInception
from models.layers import GatedFusion

class DGTANet(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, time_steps, horizon):
        super(DGTANet, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_steps = time_steps
        self.horizon = horizon

        self.graph_learner = GraphLearner(input_dim)
        self.temporal_model = DilatedInception(input_dim, output_dim, dilation_factor=2)
        self.gated_fusion = GatedFusion(output_dim, output_dim)
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        batch_size, time_steps, num_nodes, input_dim = x.shape
        adj_dynamic = self.graph_learner(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * num_nodes, time_steps, input_dim)
        x = self.temporal_model(x)
        x = x.view(batch_size, num_nodes, time_steps, self.output_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = self.gated_fusion(x)
        x = self.fc(x)
        return x
