import time
from datetime import datetime
from pathlib import Path

import torch
from dataset import Dataset
from torch.utils.data import DataLoader
from utils.image_utils import save_image
from utils.random_utils import set_seed


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

    def test_dir(self):
        pass
