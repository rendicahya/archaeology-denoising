import multiprocessing
from pathlib import Path

import click
import torch
from config import settings as conf
from models import Autoencoder, ViT
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


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


def save_image(tensor, path):
    image = tensor.cpu().clone().detach()
    image *= 255
    image = image.to(torch.uint8)
    image = Image.fromarray(image.numpy(), mode="L")

    image.save(path)


@click.command()
@click.option(
    "-c",
    "--checkpoint",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Path to the checkpoint",
)
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    required=True,
    help="Path to the input image",
)
def main(checkpoint, input):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize(conf.image_size),
            transforms.CenterCrop(conf.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # model = Autoencoder().to(device)
    model = ViT(img_size=conf.image_size, in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    Path("output").mkdir(exist_ok=True)

    if input.is_file():
        image = Image.open(input).convert("L")
        image = transform(image)
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)

        output = output.squeeze(0).squeeze(0)
        save_image(output, f"output/{model._get_name()}-{input.name}")
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
