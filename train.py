import multiprocessing
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from config import settings as conf
from dataset import Dataset
from models import Autoencoder, ViT
from timer_py import Timer
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0

    for input, target in train_loader:
        input, target = input.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(input)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)

    return train_loss


def validate(model, device, val_loader, criterion):
    model.eval()

    val_loss = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

    val_loss /= len(val_loader.dataset)

    return val_loss


def main():
    timer = Timer("Train")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transform = transforms.Compose(
        [
            transforms.Resize(conf.image_size),
            transforms.CenterCrop(conf.image_size),
            transforms.ToTensor(),
            transforms.Normalize(0, 1),
        ]
    )

    set_seed(0)
    timer.start()

    dataset = Dataset(conf.dataset.path, "list.csv", transform)
    image_size = conf.image_size
    batch_size = conf.batch_size
    n_cpu = multiprocessing.cpu_count()
    train_data, val_data, test_data = random_split(dataset, [0.7, 0.2, 0.1])
    train_loader = DataLoader(
        train_data, batch_size, shuffle=True, pin_memory=True, num_workers=n_cpu
    )
    val_loader = DataLoader(val_data, batch_size)

    # model = Autoencoder()
    model = ViT(img_size=image_size, in_channels=1, num_classes=1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = conf.epochs

    model.to(device)

    for epoch in range(1, n_epochs + 1):
        start_time = time.time()
        train_loss = train(model, device, train_loader, optimizer, criterion)
        val_loss = validate(model, device, val_loader, criterion)
        end_time = time.time()

        print(
            f"[{epoch}/{conf.epochs}], train loss: {train_loss:.6f}, val. loss: {val_loss:.6f}, time: {end_time - start_time:.2f}s"
        )

    model_filename = f"checkpoints/{model._get_name()}-{image_size}px-{n_epochs}e-{batch_size}b-{timestamp}.pth"

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_filename)
    timer.stop()


if __name__ == "__main__":
    main()
