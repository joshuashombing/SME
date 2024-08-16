import os
import shutil
import tempfile
from pathlib import Path

import cv2
from sklearn.model_selection import train_test_split

from anomalib.models.detection.spring_metal_detection import SpringMetalDetector
from anomalib.pre_processing.prepare_defect import create_anomalib_data
from anomalib.pre_processing.utils import resize_image, validate_path, generate_mask


def prepare_anomalib_from_video(video_path, output_dir, fps=2, class_name="defect", filename=None, create_mask=True):
    print(f"Processing {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / fps))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        result = model.predict(frame)
        patches = model.post_process_2(frame, result)
        for patch in patches:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            create_anomalib_data(patch, output_dir, class_name=class_name, filename=filename, create_mask=create_mask)
    cap.release()


def prepare_loop():
    data_root = Path("D:\\datasets\\datasets-SME-30-04-2024")
    output_dir = Path("D:\\datasets\\DefectDetectionDataset\\2024-05-10_08-20\\samping\\spring_sheet_metal")

    for video_path in data_root.iterdir():
        camera_side, class_name, _ = video_path.stem.split("_")
        # if class_name == "berr":
        #     class_name = "good"
        side = camera_side.split("-")[-1]

        if side != "samping":
            continue

        if class_name == "dented":
            continue

        fps = {
            "good": 2,
            "dented": 1
        }

        prepare_anomalib_from_video(
            video_path, output_dir, fps=fps.get(class_name, 2), class_name=class_name,
            create_mask=class_name != "good", filename=f"{video_path.stem}.png"
        )


def split_data(data_dir, test_size=0.7, random_state=42):
    data_dir = validate_path(data_dir)
    test_dir = data_dir / "test/good"

    if not test_dir.parent.is_dir():
        test_dir.parent.mkdir(parents=True, exist_ok=True)

    for subdir in data_dir.iterdir():
        if subdir.name in {"test", "train"}:
            continue

        shutil.move(subdir, test_dir.parent / subdir.name)

    train_set, test_set = train_test_split(
        os.listdir(test_dir), shuffle=True, test_size=test_size, random_state=random_state
    )

    train_dir = data_dir / "train/good"
    train_dir.mkdir(parents=True, exist_ok=True)

    for filename in train_set:
        file_path = test_dir / filename
        dest_path = train_dir / filename
        shutil.move(file_path, dest_path)


def split_data_v2(data_dir, test_size=0.7, random_state=42):
    data_dir = validate_path(data_dir)
    test_dir = data_dir / "test/good"

    X, y = [], []
    for filename in os.listdir(test_dir):
        filename_split = Path(filename).stem.split("_")
        if len(filename_split) == 3:
            y.append(int(filename_split[-1]))
        else:
            y.append(0)
        X.append(filename)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, test_size=test_size, random_state=random_state, stratify=y
    )

    train_dir = data_dir / "train/good"
    train_dir.mkdir(parents=True, exist_ok=True)

    for filename in X_train:
        file_path = test_dir / filename
        dest_path = train_dir / filename
        shutil.move(file_path, dest_path)


def split_data_v3(data_dir, test_size=0.7, random_state=42):
    data_dir = validate_path(data_dir)
    test_dir = data_dir / "test/good"

    X, y = [], []
    for filename in os.listdir(test_dir):
        filename_split = Path(filename).stem.split("_")
        print(filename_split)
        if len(filename_split) == 3:
            y.append(int(filename_split[0][-1]))
        else:
            y.append(0)
        X.append(filename)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, test_size=test_size, random_state=random_state, stratify=y
    )

    train_dir = data_dir / "train/good"
    train_dir.mkdir(parents=True, exist_ok=True)

    for filename in X_train:
        file_path = test_dir / filename
        dest_path = train_dir / filename
        shutil.move(file_path, dest_path)


def clear_ground_truth(data_dir):
    data_dir = validate_path(data_dir)
    test_dir = data_dir / "test"
    gt_dir = data_dir / "ground_truth"

    for gt_class_dir in gt_dir.iterdir():
        class_name = gt_class_dir.name
        test_class_dir = test_dir / class_name

        if not test_class_dir.is_dir():
            shutil.rmtree(gt_class_dir)
            continue

        filenames = set(os.listdir(gt_class_dir)) - set(os.listdir(test_class_dir))
        for filename in filenames:
            os.remove(gt_class_dir / filename)


def clear_heatmap(data_dir):
    data_dir = validate_path(data_dir)
    for img_path in data_dir.glob(r"**/*heatmap*"):
        os.remove(img_path)

    for img_path in data_dir.glob(r"**/*mask*"):
        os.remove(img_path)

    for img_path in data_dir.glob(r"**/*.png"):
        os.remove(img_path)


def generate_mask_anomalib(data_dir):
    data_dir = validate_path(data_dir)
    test_dir = data_dir / "test"

    for test_class_dir in test_dir.iterdir():
        if test_class_dir.name == "good":
            continue

        masks_dir = data_dir / "ground_truth" / test_class_dir.name
        masks_dir.mkdir(parents=True, exist_ok=True)

        for image_path in test_class_dir.iterdir():
            mask_path = masks_dir / image_path.name
            image = cv2.imread(str(image_path))
            mask = generate_mask(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            cv2.imwrite(str(mask_path), mask)


if __name__ == "__main__":
    # model = SpringMetalDetector(
    #     path="runs/segment/train/weights/best.pt",
    #     pre_processor=lambda x: resize_image(x, width=640)
    # )
    # model.build()
    # prepare_loop()
    # model = SpringMetalDetector(
    #     path="tools/yolov8.pt",
    #     pre_processor=lambda x: resize_image(x, width=640),
    #     output_patch_shape=(680, 320)
    # )
    # model.build()
    # prepare_loop()

    # split_data(Path("datasets/kamera-samping/spring_sheet_metal"))
    # split_data(Path("D:\\maftuh\\DATA\\DefectDetectionDataset\\2024-05-15_00-15\\spring_sheet_metal"))
    # split_data(Path("D:\\maftuh\\DATA\\DefectDetectionDataset\\2024-05-18_19-02-12-366456\\samping\\spring_sheet_metal"))
    # split_data(Path("D:\\maftuh\\DATA\\DefectDetectionDataset\\2024-05-18_18-54-38-681893\\atas\\spring_sheet_metal"))
    path = r"D:\maftuh\DATA\6Juni24\SpringSheetMetal\images_test\kamera-atas"
    split_data(Path(path))
    generate_mask_anomalib(Path(path))
    # split_data_v2(Path(r"D:\repos\sme-automation-inspection-internal\results\images\2024-05-31_13-09-44\kamera-atas\spring_sheet_metal"))
    # split_data_v3(Path("D:\\maftuh\\DATA\\DefectDetectionDataset\\2024-05-23_22-40-52\\samping\\spring_sheet_metal"))
    # generate_mask_anomalib("D:\\maftuh\\DATA\\DefectDetectionDataset\\2024-05-23_22-40-52\\samping\\spring_sheet_metal")
    # clear_ground_truth("D:\\maftuh\\DATA\\DefectDetectionDataset\\2024-05-22_22-05-58\\samping\\spring_sheet_metal")
