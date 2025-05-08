import torch

class Trainer:
    def __init__(self, model, loss_fn, optimizer, scaler, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device

    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)
            output = self.scaler.inverse_transform(output)
            y = self.scaler.inverse_transform(y)
            loss = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                output = self.scaler.inverse_transform(output)
                y = self.scaler.inverse_transform(y)
                loss = self.loss_fn(output, y)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def test(self, test_loader):
        self.model.eval()
        predictions = []
        truths = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                output = self.scaler.inverse_transform(output)
                y = self.scaler.inverse_transform(y)

                predictions.append(output.cpu())
                truths.append(y.cpu())

        predictions = torch.cat(predictions, dim=0)
        truths = torch.cat(truths, dim=0)
        return predictions, truths
