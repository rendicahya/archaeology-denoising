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
    # model = ViT(img_size=conf.image_size, in_channels=1, num_classes=1).to(device)
    # model.load_state_dict(torch.load(checkpoint, map_location=device))

    if input.is_file():
        image = Image.open(input).convert("L")
        tester.test_file(image, input.name)
    else:
        dataset = Dataset(input, transform)
        batch_size = conf.batch_size
        n_cpu = multiprocessing.cpu_count()
        loader = DataLoader(dataset, batch_size, pin_memory=True, num_workers=n_cpu)
        n = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                output = model(batch)

                for image in output:
                    image = image.squeeze(0)

                    save_image(image, f"output/{model._get_name()}-{n}.jpg")
                    n += 1


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        self.paths = [f for f in Path(path).iterdir() if f.is_file()]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("L")
        image = self.transform(image)

        return image


if __name__ == "__main__":
    main()
