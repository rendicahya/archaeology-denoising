import multiprocessing

from models.vit import ViT
from torch import nn, optim
from torchvision import transforms

image_size = 600
transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(0, 1),
    ]
)
dataset_path = "data/synthetic-v1"
split_ratio = [0.7, 0.2, 0.1]
shuffle_train = True
random_seed = 0
n_epochs = 50
batch_size = 54
learning_rate = 1e-4
model = ViT(image_size=image_size, in_channels=1, num_classes=1)
criterion = nn.MSELoss
optimizer = optim.Adam
num_workers = multiprocessing.cpu_count()
