from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import time


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


def save_frame(video_path: Path, save_dir: Path, gap=10):
    save_path = save_dir / video_path.stem
    save_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.release()
            break

        frame = resize_image(frame, width=640)

        new_filename = f"{idx}_{str(time.time()).split('.')[0]}.png"

        if idx == 0:
            cv2.imwrite(f"{save_path}/{new_filename}", frame)
        else:
            if idx % gap == 0:
                cv2.imwrite(f"{save_path}/{new_filename}", frame)

        idx += 1


def main():
    data_dir = Path("D:\\MV-CAMERA")
    output_dir = Path("D:\\DATA\\SME\\ObjectDetectionDataset")

    for path in tqdm(data_dir.glob('**/*.avi')):
        save_frame(path, output_dir, gap=5)


if __name__ == "__main__":
    main()
