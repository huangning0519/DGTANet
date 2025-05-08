import torch

def mae(preds, labels):
    return torch.mean(torch.abs(preds - labels)).item()

def mape(preds, labels):
    mask = labels != 0
    return torch.mean(torch.abs((preds[mask] - labels[mask]) / labels[mask])).item()

def rmse(preds, labels):
    return torch.sqrt(torch.mean((preds - labels) ** 2)).item()
