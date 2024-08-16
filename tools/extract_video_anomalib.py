import re
from pathlib import Path

import cv2

from anomalib.models.detection.defect_detection_v1 import DefectPreprocessor
from anomalib.models.detection.spring_metal_detection import SpringMetalDetector
from anomalib.pre_processing.prepare_defect import create_anomalib_data
from anomalib.pre_processing.utils import get_now_str, resize_image, draw_bbox
from config import DETECTOR_PARAMS_TOP_CAMERA, DETECTOR_PARAMS_SIDE_CAMERA, PREPROCESSOR_PARAMS_TOP_CAMERA, \
    PREPROCESSOR_PARAMS_SIDE_CAMERA

side_names = {"atas", "samping"}
class_names = {"good", "dented", "shift", "berr", "double"}


def draw_result(frame, boxes, object_ids):
    frame = resize_image(frame, width=640)

    if len(boxes) == 0:
        return frame

    for box, object_id in zip(boxes, object_ids):
        draw_bbox(frame, box=box, label=f"Id {object_id}", score=None, color=(255, 255, 255))

    return frame


def get_side_class_map(path: Path):
    strings = re.split(r'[-_/\\ ]', path.resolve().as_posix().lower())

    for side in side_names:
        if side in strings:
            break
    else:
        side = None  # Set side to None if no match found

    for class_name in class_names:
        if class_name in strings:
            break
    else:
        class_name = None  # Set class_name to None if no match found

    if side not in side_names:
        raise ValueError(f"Invalid side name in path '{path}'. Expected one of {side_names}.")

    if class_name not in class_names:
        raise ValueError(f"Invalid class name in path '{path}'. Expected one of {class_names}.")

    return side, class_name


def predict_video(
        detector: SpringMetalDetector,
        preprocessor: DefectPreprocessor,
        video_path: Path,
        video_version: str,
        class_name: str,
        output_dir: Path
):
    cap = cv2.VideoCapture(str(video_path))
    # print(f"Processing {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        result = detector.predict(frame)

        patches_map = preprocessor.object_post_process(frame, result)

        for position, object_id, patch in patches_map:
            create_anomalib_data(
                cv2.cvtColor(patch, cv2.COLOR_BGR2RGB),
                output_dir,
                class_name=class_name,
                filename=f"{video_version}_{video_path.stem}_object_{object_id}.jpg",
                create_mask=class_name != "good"
            )

        frame = draw_result(frame, boxes=result["boxes_n"], object_ids=result["track_ids"])

        cv2.imshow(video_path.name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    output_root_dir = Path(f"D:\\maftuh\\DATA\\DefectDetectionDataset\\{get_now_str(microsecond=False)}")
    data_dirs = [
        Path("D:\\maftuh\\DATA\\SME-VIDEO-Dataset"),
        # Path("D:\\maftuh\\DATA\\MV-CAMERA"),
        Path("D:\\maftuh\\DATA\\16Apr24-sme-spring sheet"),
        Path("D:\\maftuh\\Projects\\SME\\anomalib\\datasets"),
        Path("D:\\maftuh\\DATA\\2024-05-15_13-00"),
        Path("D:\\maftuh\\DATA\\datasets-130524"),
    ]

    detector_params = {
        "atas": DETECTOR_PARAMS_TOP_CAMERA,
        "samping": DETECTOR_PARAMS_SIDE_CAMERA
    }
    preprocessor_params = {
        "atas": PREPROCESSOR_PARAMS_TOP_CAMERA,
        "samping": PREPROCESSOR_PARAMS_SIDE_CAMERA
    }

    print("Output directory", output_root_dir)

    for i, data_dir in enumerate(data_dirs):
        for video_path in data_dir.glob("**/*.avi"):
            # try:
            side, class_name = get_side_class_map(video_path)
            print(f"Processing video: Side {side}, Class {class_name}")
            print("Path:", video_path)

            # if side != "samping":
            #     continue

            output_dir = output_root_dir / f"{side}/spring_sheet_metal"
            output_dir.mkdir(parents=True, exist_ok=True)

            detector = SpringMetalDetector(**detector_params[side]).build()
            # params = preprocessor_params[side].copy()
            # if i == 1 and side == "atas":
            # params["target_shape"] = (540, 444)
            preprocessor = DefectPreprocessor(**preprocessor_params[side])
            video_version = f"v{i}"

            # if side == "atas" and i == 1:
            #     continue

            # if side == "atas":
            #     if class_name == "berr":
            #         class_name = "good"
            # else:
            if class_name == "dented" and side == "samping":
                continue

            if class_name == "double stamp":
                class_name = "berr"

            # print(f"Processing video: Side {side}, Class {class_name}")
            predict_video(detector, preprocessor, video_path, video_version, class_name, output_dir=output_dir)
            # except ValueError as e:
            #     print("Error", e)


if __name__ == "__main__":
    main()
    # test = get_side_class_map(Path("D:\\maftuh\\DATA\\SME-VIDEO-Dataset\\berr\\samping\\Video_20240420183439333.avi"))
    # test = get_side_class_map(Path("D:\\maftuh\\Projects\\SME\\anomalib\\datasets\\kamera-atas_berr_1714379252-8252501.avi"))
    # print(test)
