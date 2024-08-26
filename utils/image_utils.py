import torch
from PIL import Image


def save_image(tensor, path):
    image = tensor.cpu().clone().detach()
    image *= 255
    image = image.to(torch.uint8)
    image = Image.fromarray(image.numpy(), mode="L")

    image.save(path)
