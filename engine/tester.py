from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.image_utils import save_image
from utils.random_utils import set_seed


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        self.paths = [f for f in Path(path).iterdir() if f.is_file()]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image

        path = self.paths[idx]
        image = Image.open(path).convert("L")
        image = self.transform(image)

        return image


class Tester:
    def __init__(self, config, checkpoint):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = checkpoint
        self.config = config

        config.model.to(self.device)
        config.model.load_state_dict(torch.load(checkpoint, map_location=self.device))

        if config.random_seed:
            set_seed(config.random_seed)

    def test_file(self, input, filename):
        input = self.config.transform(input)
        input = input.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.config.model(input)

        output = output.squeeze(0).squeeze(0)
        output_path = Path("output") / self.checkpoint.stem / filename

        output_path.parent.mkdir(exist_ok=True)
        save_image(output, output_path)

    def test_dir(self, input):
        dataset = Dataset(input, self.config.transform)
        data_loader = DataLoader(
            dataset,
            self.config.batch_size,
            pin_memory=True,
            num_workers=self.config.num_workers,
        )
        n = 0
        output_dir = Path("output") / self.checkpoint.stem / input.name
        bar = tqdm(total=len(dataset))

        output_dir.mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                output = self.config.model(batch)

                for image in output:
                    image = image.squeeze(0)
                    output_path = output_dir / f"{n}.jpg"
                    n += 1

                    save_image(image, output_path)
                    bar.update(1)
