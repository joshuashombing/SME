from pathlib import Path
from typing import Union

import cv2

from anomalib.pre_processing.utils import validate_path, get_now_str, generate_unique_filename, generate_mask


def create_anomalib_data(image, output_dir: Union[Path, str], class_name, filename=None, create_mask=False):
    path = validate_path(output_dir)
    images_dir = path / "test" / class_name
    images_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{get_now_str()}.jpg"

    image_path = images_dir / filename
    image_path, _ = generate_unique_filename(image_path)
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if create_mask:
        masks_dir = path / "ground_truth" / class_name
        masks_dir.mkdir(parents=True, exist_ok=True)
        mask_path = masks_dir / image_path.name
        mask = generate_mask(image)
        cv2.imwrite(str(mask_path), mask)
