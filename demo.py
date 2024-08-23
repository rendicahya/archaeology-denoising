from pathlib import Path

import click
import torch
from config import settings as conf
from models import Autoencoder, ViT
from PIL import Image
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
    image = image.squeeze(0).squeeze(0)
    image *= 255
    image = image.to(torch.uint8)
    image = Image.fromarray(image.numpy(), mode="L")

    image.save(path)


@cli.command()
@click.option(
    "-i",
    "--image",
    type=click.Path(exists=True),
    required=True,
    file_okay=True,
    dir_okay=True,
    help="Path to the image",
)
@click.option(
    "-m",
    "--model",
    type=click.Path(exists=True),
    required=True,
    file_okay=True,
    dir_okay=False,
    help="Path to the model",
)
def main():
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

    model_path = "checkpoints/ViT-600-20240822_182409.pth"
    image_path = Path("../../datasets/arch_denoising/noise/image-002.jpg")
    image = Image.open(image_path).convert("L")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    with torch.no_grad():
        output = model(image)

    save_image(output, f"output/{model._get_name()}-{image_path.name}")


if __name__ == "__main__":
    main()
