import multiprocessing
from pathlib import Path

import click
import torch
from config import settings as conf
from engine.tester import Tester
from PIL import Image
from torch.utils.data import DataLoader
from utils.config_utils import load_config


def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)

    print(f"Test loss: {test_loss:.6f}")


@click.command()
@click.argument(
    "config-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.argument(
    "checkpoint",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
)
@click.argument(
    "input",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
def main(config_path, checkpoint, input):
    config = load_config(config_path)
    tester = Tester(config, checkpoint)

    if input.is_file():
        image = Image.open(input).convert("L")
        tester.test_file(image, input.name)
    else:
        tester.test_dir(input)


if __name__ == "__main__":
    main()
