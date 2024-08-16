import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_parser() -> ArgumentParser:
    """Get parser for split_data function.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser(description="Split data into train and test sets.")
    parser.add_argument("data_dir", type=Path, help="Path to the directory containing the data.")
    parser.add_argument("--test_size", type=float, default=0.5,
                        help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Controls the shuffling applied to the data before splitting.")
    parser.add_argument("--create_mask", action="store_true", help="Whether to create a mask or not.")
    return parser


def validate_path(path: Path | str):
    return Path(path) if isinstance(path, str) else path


def _create_mask(img, fill=255):
    assert img is not None, "file could not be read, check with os.path.exists()"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)
    bin_img[bin_img != 0] = 126
    bin_img[bin_img == 0] = fill
    bin_img[bin_img == 126] = 0
    return bin_img.astype(np.uint8)


def _generate_mask(class_dir: Path | str):
    class_dir = validate_path(class_dir)
    test_dir = class_dir / "test"

    if not test_dir.is_dir():
        return

    gt_dir = class_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    for path in test_dir.iterdir():
        if path.name.lower() == "good":
            continue

        class_gt_dir = gt_dir / path.name
        class_gt_dir.mkdir(parents=True, exist_ok=True)

        for file_path in path.iterdir():
            if not file_path.is_file():
                continue

            if "mask" in file_path.stem:
                continue

            mask_path = file_path.parent / f"{file_path.stem}_mask.png"
            out_path = class_gt_dir / file_path.name

            if mask_path.is_file():
                shutil.move(mask_path, out_path)
                continue

            img = cv2.imread(str(file_path))
            mask = _create_mask(img, fill=255)

            if out_path.is_file():
                os.remove(out_path)

            cv2.imwrite(str(out_path), mask)


def _train_test_split(class_dir: Path | str, test_size=0.5, random_state=42):
    class_dir = validate_path(class_dir)

    good_dir = class_dir / "good"
    list_dir = [path.name for path in good_dir.iterdir() if path.is_file() and "mask" not in path.name]

    if len(list_dir) == 0:
        print(
            f"Good directory is empty, please check at {good_dir}"
        )
        return

    train_set, test_set = train_test_split(
        list_dir, shuffle=True, test_size=test_size, random_state=random_state
    )

    train_dir = class_dir / "train" / good_dir.name
    test_dir = class_dir / "test" / good_dir.name
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for filename in list_dir:
        output_dir = train_dir if filename in train_set else test_dir
        src_path = good_dir / filename
        dst_path = output_dir / filename
        shutil.move(src_path, dst_path)

    # assert len(os.listdir(good_dir)) == 0, (
    #     f"Good directory still not empty. "
    #     f"Got len {len(os.listdir(good_dir))}, "
    #     f"please check at {good_dir}"
    # )

    shutil.rmtree(good_dir)

    for path in class_dir.iterdir():

        if path.name in {"train", "test"}:
            continue

        shutil.move(path, test_dir.parent / path.name)


def split_data(data_dir: Path | str, test_size=0.5, random_state=42, create_mask=False):
    data_dir = validate_path(data_dir)

    for class_dir in tqdm(data_dir.iterdir(), desc=f"Splitting data for {data_dir.name}"):
        if not class_dir.is_dir():
            continue
        _train_test_split(class_dir, test_size=test_size, random_state=random_state)

        if create_mask:
            _generate_mask(class_dir)


if __name__ == "__main__":
    args = get_parser().parse_args()
    split_data(args.data_dir, args.test_size, args.random_state, args.create_mask)
