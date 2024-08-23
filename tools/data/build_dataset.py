import sys

sys.path.append(".")

import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from config import settings as S
from tqdm import tqdm


def generate_spots(img):
    result = img.copy()
    num_spots = 50
    height, width = img.shape[:2]
    spot_centers = np.random.rand(num_spots, 2) * [width, height]
    spot_centers = spot_centers.astype(int)
    spot_radii = np.random.randint(10, 40, num_spots)
    alphas = np.random.uniform(0.2, 0.6, num_spots)
    num_vertices = np.random.randint(5, 40, num_spots)

    for center, radius, alpha, vertices in zip(
        spot_centers, spot_radii, alphas, num_vertices
    ):
        angles = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
        x_coords = (
            center[0] + radius * np.cos(angles) + np.random.uniform(-5, 5, vertices)
        )
        y_coords = (
            center[1] + radius * np.sin(angles) + np.random.uniform(-5, 5, vertices)
        )
        pts = np.stack((x_coords, y_coords), axis=-1).astype(np.int32)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], (255))

        blurred_mask = cv2.GaussianBlur(mask, (21, 21), 0)
        normalized_mask = blurred_mask.astype(np.float32) / 255.0

        result[:, :] = (1 - normalized_mask * alpha) * result[
            :, :
        ] + normalized_mask * alpha * 0

    return result


def main():
    dataset_path = Path(S.dataset.path)
    transforms = [
        A.NoOp(),
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        A.Perspective(p=1),
        A.Morphological(p=1),
        A.OpticalDistortion(p=1),
        A.ElasticTransform(p=1),
        A.PiecewiseAffine(p=1),
        A.Transpose(p=1),
        A.Affine(p=1),
    ]
    n_src_image = sum(1 for f in (dataset_path / "src").iterdir() if f.is_file())
    n_generate = S.dataset.generate
    bar = tqdm(total=n_src_image * len(transforms) * n_generate)
    file_list = {"clean": [], "noise": []}
    count = 0

    random.seed(0)
    (dataset_path / "clean").mkdir(exist_ok=True)
    (dataset_path / "noise").mkdir(exist_ok=True)

    for file in (dataset_path / "src").iterdir():
        image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)

        for t in transforms:
            transformed = A.Compose([t])(image=image)["image"]

            for i in range(n_generate):
                noise_image = generate_spots(transformed)
                count += 1

                clean_image_path = dataset_path / f"clean/image-{count:03}.jpg"
                noise_image_path = dataset_path / f"noise/image-{count:03}.jpg"

                cv2.imwrite(str(clean_image_path), transformed)
                cv2.imwrite(str(noise_image_path), noise_image)

                file_list["clean"].append(clean_image_path.relative_to(dataset_path))
                file_list["noise"].append(noise_image_path.relative_to(dataset_path))

                bar.update(1)

    pd.DataFrame(file_list).to_csv(dataset_path / "list.csv", index=False)


if __name__ == "__main__":
    main()
