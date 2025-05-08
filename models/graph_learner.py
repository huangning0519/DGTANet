import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(GraphLearner, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, time_steps, num_nodes, input_dim = x.shape
        x_mean = x.mean(dim=1)
        h = F.relu(self.linear1(x_mean))
        h = self.linear2(h)
        adj = torch.matmul(h, h.transpose(1, 2))
        adj = F.relu(adj)
        adj_sum = adj.sum(dim=-1, keepdim=True) + 1e-6
        adj_normalized = adj / adj_sum
        return adj_normalized
