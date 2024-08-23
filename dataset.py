from pathlib import Path

import pandas as pd
import torch.utils
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, file_list, transform):
        self.dataset_dir = Path(dataset_dir)
        self.data_frame = pd.read_csv(self.dataset_dir / file_list, engine="pyarrow")
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        clean_image_path = self.dataset_dir / str(self.data_frame.iloc[idx, 0])
        noise_image_path = self.dataset_dir / str(self.data_frame.iloc[idx, 1])

        clean_image = Image.open(clean_image_path).convert("L")
        noise_image = Image.open(noise_image_path).convert("L")

        clean_image = self.transform(clean_image)
        noise_image = self.transform(noise_image)

        return noise_image, clean_image
