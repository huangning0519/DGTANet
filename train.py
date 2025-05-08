import os
import torch
import torch.optim as optim
import numpy as np
import random
import yaml
from models.dgtanet import DGTANet
from data.dataloader import get_dataloader
from utils.engine import Trainer
from utils.utils import StandardScaler
from utils.metrics import mae, mape, rmse

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, val_loader, test_loader, scaler = get_dataloader(
    data_dir=config['data_dir'],
    batch_size=config['batch_size'],
    time_steps=config['time_steps'],
    horizon=config['horizon'],
    input_dim=config['input_dim'],
    output_dim=config['output_dim']
)

model = DGTANet(
    num_nodes=config['num_nodes'],
    input_dim=config['input_dim'],
    output_dim=config['output_dim'],
    time_steps=config['time_steps'],
    horizon=config['horizon']
).to(device)

optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
loss_fn = torch.nn.SmoothL1Loss()

trainer = Trainer(model, loss_fn, optimizer, scaler, device)

for epoch in range(1, config['epochs'] + 1):
    print(f"Epoch {epoch}/{config['epochs']}")
    train_loss = trainer.train(train_loader)
    val_loss = trainer.validate(val_loader)
    print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

test_predictions, test_truths = trainer.test(test_loader)
mae_score = mae(test_predictions, test_truths)
mape_score = mape(test_predictions, test_truths)
rmse_score = rmse(test_predictions, test_truths)
print(f"Test MAE: {mae_score:.4f}, Test MAPE: {mape_score:.4f}, Test RMSE: {rmse_score:.4f}")

os.makedirs('saved_models', exist_ok=True)
torch.save(model.state_dict(), 'saved_models/dgtanet.pth')
