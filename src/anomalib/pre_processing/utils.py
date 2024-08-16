import shutil
import tempfile
import time
from typing import Union

from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


def validate_path(path: Union[Path, str]):
    return Path(path) if isinstance(path, str) else path


def generate_unique_filename(output_path: Path) -> tuple[Path, int]:
    index = 1
    base_filename = output_path.stem
    while output_path.exists():
        filename = f"{base_filename}_{index}{output_path.suffix}"
        output_path = output_path.parent / filename
        index += 1
    return output_path, index


def get_now_str(timestamp: Union[float, None] = None, microsecond=True) -> str:
    """Standard format for datetimes is defined here."""

    if timestamp is None:
        timestamp = datetime.fromtimestamp(time.time())

    string = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

    if microsecond:
        string += f"-{timestamp.microsecond}"

    return string


def resize_shape(source_shape, width=None, height=None):
    (h, w) = source_shape

    if width is None and height is None:
        return h, w

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        shape = (height, int(w * r))

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        shape = (int(h * r), width)

    return shape


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA) -> np.ndarray:
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    h, w = resize_shape(source_shape=image.shape[:2], width=width, height=height)

    # resize the image
    resized = cv2.resize(image, (w, h), interpolation=inter)

    # return the resized image
    return resized


def generate_mask(image):
    assert image is not None, "file could not be read, check with os.path.exists()"
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)
    return bin_img.astype(np.uint8)


def expand_bbox(bbox, percentage=0.01):
    if isinstance(percentage, float):
        percentage = [percentage, percentage]

    is_normalized = np.all(np.array(bbox) <= 1)

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Calculate the expansion amount
    width_expansion = width * percentage[0]
    height_expansion = height * percentage[1]

    new_x1 = x1 - width_expansion
    new_x2 = x2 + width_expansion
    new_y1 = y1 - height_expansion
    new_y2 = y2 + height_expansion

    if not is_normalized:
        new_x1, new_y1 = int(new_x1), int(new_y1)
        new_x2, new_y2 = int(new_x2), int(new_y2)

    return new_x1, new_y1, new_x2, new_y2


def draw_bbox(frame, box, label=None, score=None, color=(255, 255, 255)):
    height, width = frame.shape[:2]

    # Calculate thickness dynamically based on the image size
    thickness = max(1, min(height, width) // 200)

    x1, y1, x2, y2 = expand_bbox(box, 0.05)
    x1 = int(x1 * width)
    y1 = int(y1 * height)
    x2 = int(x2 * width)
    y2 = int(y2 * height)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    text = ""
    if label is not None:
        text += label

    if score is not None:
        if text:
            text += ": "
        text += f"{score:.2f}"

    if text:
        font_scale = 0.4  # Adjust font scale based on the image size
        font_thickness = max(1, thickness // 2)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                              font_thickness)

        # Draw a filled rectangle as background for the text
        cv2.rectangle(frame, (x1, y1 - text_height - baseline - 4), (x1 + text_width + 8, y1), color,
                      thickness=cv2.FILLED)

        # Draw the text on top of the rectangle
        cv2.putText(frame, text, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

    return frame


def draw_line(frame, distances, color=(255, 255, 255), thickness=2):
    height, width = frame.shape[:2]
    wn, hn = distances

    x1n = max(0, 0.5 - wn)
    x2n = min(1, 0.5 + wn)

    y1n = max(0, 0.5 - hn)
    y2n = min(1, 0.5 + hn)

    if x1n != 0:
        x1 = int(x1n * width)
        cv2.line(frame, (x1, 0), (x1, height), color, thickness=thickness)

    if x2n != 1:
        x2 = int(x2n * width)
        cv2.line(frame, (x2, 0), (x2, height), color, thickness=thickness)

    if y1n != 0:
        y1 = int(y1n * height)
        cv2.line(frame, (0, y1), (width, y1), color, thickness=thickness)

    if y2n != 1:
        y2 = int(y2n * height)
        cv2.line(frame, (0, y2), (width, y2), color, thickness=thickness)

    return frame


def add_background(foreground_image, target_shape):
    """
    Replace the background of an image with a black background and place the cropped foreground image in the center.

    Parameters:
        foreground_image (numpy.ndarray): The foreground image.
        target_shape (tuple): Tuple containing (target_width, target_height) of the output image.

    Returns:
        numpy.ndarray: The image with the black background and the cropped foreground image placed in the center.
    """
    # Get dimensions of the foreground image
    fg_height, fg_width, _ = foreground_image.shape

    # Calculate the cropping region to fit within the target shape
    crop_x1 = max(0, (fg_width - target_shape[0]) // 2)
    crop_x2 = min(fg_width, (fg_width + target_shape[0]) // 2)
    crop_y1 = max(0, (fg_height - target_shape[1]) // 2)
    crop_y2 = min(fg_height, (fg_height + target_shape[1]) // 2)

    # Crop the foreground image
    cropped_foreground = foreground_image[crop_y1:crop_y2, crop_x1:crop_x2]

    # Create a black background image
    background_image = np.zeros((target_shape[1], target_shape[0], 3), dtype=np.uint8)

    # Get dimensions of the cropped foreground image
    fg_height, fg_width, _ = cropped_foreground.shape

    # Calculate coordinates to place the cropped foreground image in the center of the black background image
    x_offset = (target_shape[0] - fg_width) // 2
    y_offset = (target_shape[1] - fg_height) // 2

    # Put the cropped foreground image onto the black background image
    background_image[y_offset:y_offset + fg_height, x_offset:x_offset + fg_width] = cropped_foreground

    return background_image


def find_class_dirs(data_dir: Path):
    class_names = {"kamera-atas", "kamera-samping"}

    if data_dir.name in class_names:
        return [data_dir]

    class_dirs = []
    for subdir in data_dir.iterdir():
        if subdir.name in class_names:
            class_dirs.append(subdir)

    if len(class_dirs) == 0:
        print(
            f"Error: No directories matching class names found in the specified "
            f"data directory. Please create with the name {class_names}"
        )
        return []

    return class_dirs


def process_data_directory(data_dir, process_func, output_dir=None):
    try:
        data_dir = validate_path(data_dir)
        output_dir = validate_path(output_dir) if output_dir is not None else None
        # Create a temporary directory to perform operations safely
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root_dir = Path(temp_dir) / data_dir.name
            shutil.copytree(data_dir, temp_root_dir)
            process_func(temp_root_dir)

            if output_dir is not None:
                target_dir = output_dir
                # if (target_dir / temp_root_dir.name).is_dir():
                #     print(target_dir / temp_root_dir.name)
                #     print(temp_root_dir.parent / f"{temp_root_dir.name}_{get_now_str(microsecond=False)}")
                #     temp_root_dir.rename(temp_root_dir.parent / f"{temp_root_dir.name}_{get_now_str(microsecond=False)}")
                #     print(temp_root_dir)
                shutil.move(temp_root_dir, target_dir)
                (target_dir / f"{temp_root_dir.name}").rename(
                    target_dir / f"{temp_root_dir.name}_{data_dir.parent.name}")
            else:
                target_dir = data_dir
                data_dir.rename(data_dir.parent / f"{data_dir.name}_original")
                shutil.move(temp_root_dir, target_dir)
    except FileNotFoundError as e:
        print(f"Error: {e.strerror} - {e.filename}")
    except PermissionError as e:
        print(f"Error: {e.strerror} - {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    finally:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
