import time
from datetime import datetime
from pathlib import Path

import torch
from dataset import Dataset
from timer_py import Timer
from torch.utils.data import DataLoader, random_split
from utils.random_utils import set_seed


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.random_seed:
            set_seed(config.random_seed)

    def train_epoch(self, train_loader, optimizer, criterion):
        self.config.model.train()
        train_loss = 0

        for input, target in train_loader:
            input, target = input.to(self.device), target.to(self.device)

            optimizer.zero_grad()

            output = self.config.model(input)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)

        return train_loss

    def validate_epoch(self, model, val_loader, criterion):
        model.eval()

        val_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)

        return val_loss

    def train(self):
        timer = Timer(self.config.model._get_name())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        optimizer = self.config.optimizer(
            self.config.model.parameters(), lr=self.config.learning_rate
        )
        criterion = self.config.criterion()
        dataset = Dataset(self.config.dataset_path, "list.csv", self.config.transform)
        train_data, val_data, test_data = random_split(dataset, self.config.split_ratio)
        train_loader = DataLoader(
            train_data,
            self.config.batch_size,
            shuffle=self.config.shuffle_train,
            pin_memory=True,
            num_workers=self.config.num_workers,
        )
        val_loader = DataLoader(val_data, self.config.batch_size)

        self.config.model.to(self.device)
        timer.start()

        for epoch in range(1, self.config.n_epochs + 1):
            start_time = time.time()
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.validate_epoch(self.config.model, val_loader, criterion)
            elapsed = f"{time.time() - start_time:.2f}"

            print(
                f"[{epoch}/{self.config.n_epochs}], train loss: {train_loss:.6f}, val. loss: {val_loss:.6f}, time: {elapsed}s"
            )

        checkpoint_path = Path(f"checkpoints/{self.config.config_name}/{timestamp}.pth")

        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(self.config.model.state_dict(), checkpoint_path)
        timer.stop()
