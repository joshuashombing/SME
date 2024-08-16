import time
from datetime import datetime
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from anomalib.models.detection.utils import expand_bbox
from deskew import determine_skew
from omegaconf import OmegaConf
from skimage.transform import rotate

from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.models.detection.spring_metal_detection import SpringMetalDetector, resize_image


def point_within_circle(point, center, radius):
    x, y = point
    center_x, center_y = center
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return distance <= radius


def transform_bbox(result, img_shape):
    height, width = img_shape[:2]

    cls, conf, xyxyn = result
    x1, y1, x2, y2 = xyxyn

    x1 = int(x1 * width)
    y1 = int(y1 * height)
    x2 = int(x2 * width)
    y2 = int(y2 * height)
    return x1, y1, x2, y2


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


color_map = {
    "default": (255, 255, 255),
    "good": (0, 255, 0),
    "defect": (0, 0, 255)
}


def draw_bbox(frame, result: dict):
    if result is None:
        return frame

    height, width = frame.shape[:2]

    label = result.get("label")
    score = result.get("score")
    target_names = result.get("target_names")
    xyxyn = result["bbox"]

    if target_names is not None and label is not None:
        label_name = target_names[label]
    else:
        label_name = None

    color = color_map.get(str(label_name).lower(), color_map["default"])

    x1, y1, x2, y2 = expand_bbox(xyxyn, 0.05)
    x1 = int(x1 * width)
    y1 = int(y1 * height)
    x2 = int(x2 * width)
    y2 = int(y2 * height)

    thickness = 10
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if label_name is not None and score is not None:
        text = f"{label_name}: {score:.2f}"
        cv2.putText(frame, text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, color, thickness)

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


class SpringMetalInspector:
    def __init__(self):
        self.project_dir = Path(__file__).parent.parent.parent
        self.configs_dir = self.project_dir / "configs"
        self.class_names = ["Good", "Defect"]
        self.device = "cpu"
        # self.device = torch.device(
        #     "cuda:0" if torch.cuda.is_available() else "cpu"
        # )
        self.models = {}
        self.transforms = {}
        self.post_process_detections = {
            "kamera-atas": lambda img_: self.post_process_detection(
                img_, target_shape=(680, 560), rotate_image=True
            ),
            "kamera-samping": lambda img_: self.post_process_detection(
                img_, target_shape=(640, 240), rotate_image=False
            )
        }

        self.output_dir = Path("./data")
        self.detector_path = self.project_dir / "tools/yolov8.pt"
    def _build_model(self) -> None:
        for config_path in self.configs_dir.iterdir():
            if config_path.suffix != ".yaml":
                continue

            model_name = config_path.stem
            self.models[model_name] = {}
            config = OmegaConf.load(config_path)
            config.visualization.show_images = False
            config.visualization.save_images = False
            config.visualization.log_images = False
            config.visualization.image_save_path = None

            config.trainer.resume_from_checkpoint = str(
                Path(self.project_dir / config.project.path) / "weights/lightning/model.ckpt")
            weight_path = config.trainer.resume_from_checkpoint

            self.models[model_name] = get_model(config)
            self.models[model_name].load_state_dict(torch.load(weight_path, map_location=self.device)["state_dict"])
            self.models[model_name].to(self.device)
            self.models[model_name].eval()

            # get the transforms
            transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
            image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
            center_crop = config.dataset.get("center_crop")

            if center_crop is not None:
                center_crop = tuple(center_crop)

            normalization = InputNormalizationMethod(config.dataset.normalization)
            self.transforms[model_name] = get_transforms(
                config=transform_config,
                image_size=image_size,
                center_crop=center_crop,
                normalization=normalization
            )

        self.roi_area_size = [0.2, 0.5]
        self.detector = SpringMetalDetector(
            path=self.detector_path,
            max_num_detection=1,
            distance_thresholds=self.roi_area_size,
            device=self.device
        )

    def post_process_detection(self, image, target_shape=(680, 560), rotate_image=False):

        if rotate_image:
            angle = determine_skew(image)
            image = (rotate(image, angle, resize=False) * 255).astype(np.uint8)

        image = add_background(image, target_shape)
        return image

    def detect(self, image: np.ndarray, model_name):
        results = self.detector.detect(image)

        frames = []
        for i, result in enumerate(results):
            x1, y1, x2, y2 = transform_bbox(result, image.shape)

            cropped_frame = image[y1:y2, x1:x2]
            cropped_frame = self.post_process_detections[model_name](cropped_frame)
            # output_dir = self.output_dir / model_name / "shift"
            # output_dir.mkdir(parents=True, exist_ok=True)
            # cv2.imwrite(str(output_dir / f"{str(time.time()).replace('.', '_')}.jpg"), cropped_frame)
            frames.append(cropped_frame)

        return results, frames

    def pre_process_defect(self, frames, transform):
        collected_batch = None
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch = transform(image=frame)
            batch["image"] = batch["image"].unsqueeze(0)

            if collected_batch is None:
                collected_batch = batch.copy()
            else:
                collected_batch["image"] = torch.cat((collected_batch["image"], batch["image"]), dim=0)

        collected_batch["image"] = collected_batch["image"].to(self.device)

        return collected_batch

    def predict(self, image: np.ndarray, model_name):
        start_time = datetime.now()
        ori_image = image.copy()

        detection_result, frames = self.detect(image, model_name)
        # if len(frames) > 0:
        #     output_dir = self.output_dir / model_name / "shift_ori"
        #     output_dir.mkdir(parents=True, exist_ok=True)
        #     cv2.imwrite(str(output_dir / f"{str(time.time()).replace('.', '_')}.jpg"), ori_image)
        if len(frames) > 0:
            selected_frames = [frames.pop(0)]
            selected_detection = [detection_result.pop(0)]
            batch = self.pre_process_defect(selected_frames, self.transforms[model_name])
            with torch.no_grad():
                defect_result = self.models[model_name].predict_step(batch, 0)

            pred_scores = defect_result["pred_scores"].cpu().numpy()
            pred_labels = defect_result["pred_labels"].cpu().numpy()

            for detection, score, label in zip(selected_detection, pred_scores, pred_labels):
                _, _, xyxyn = detection
                result = {
                    "label": int(label),
                    "score": score,
                    "bbox": xyxyn,
                    "target_names": ["Good", "Defect"]
                }
                ori_image = draw_bbox(ori_image, result)
            delta_time = datetime.now() - start_time
            print("Inference time:", delta_time)

        for cls, conf, xyxyn in detection_result:
            result = {
                "label": None,
                "score": None,
                "bbox": xyxyn
            }
            ori_image = draw_bbox(ori_image, result)

        ori_image = draw_line(ori_image, self.roi_area_size)
        return ori_image


def show_image(img):
    if img is not None:
        cv2.imshow("Image", img)


def get_data_path_generator():
    data_root1 = Path("D:\\maftuh\\DATA\\SME-VIDEO-Dataset")
    # data_root2 = Path("D:\\maftuh\\DATA\\MV-CAMERA\\MV-CE120-10GM (00DA2265775)")
    # data_root3 = Path("D:\\maftuh\\DATA\\MV-CAMERA\\MV-CE120-10GM (00DA2265758)")
    data_root4 = Path("D:\\maftuh\\DATA\\16Apr24-sme-spring sheet")

    data_paths = {
        "atas": [
            data_root1.glob("**/atas/*.avi"),
            # data_root2.glob("**/*.avi"),
            # data_root4.glob("**/*atas*.avi"),
        ],
        "samping": [
            data_root1.glob("**/samping/*.avi"),
            # data_root3.glob("**/*.avi"),
            # data_root4.glob("**/*samping*.avi"),
        ]
    }
    return data_paths


def predict_video(model: SpringMetalInspector, video_path: Path, camera_name):
    cap = cv2.VideoCapture(str(video_path))
    print(f"Processing {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            cap.release()
            break

        frame = model.predict(frame, camera_name)
        frame = resize_image(frame, width=1024)

        cv2.imshow(video_path.name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def stream_parallel(camera_ids, process=lambda i, x: x):
    cap = [cv2.VideoCapture(i) for i in camera_ids]

    window_titles = [f"camera {i}" for i in range(len(camera_ids))]
    frames = [None] * len(camera_ids)
    processed_frames = [None] * len(camera_ids)
    ret = [None] * len(camera_ids)

    while True:

        for i, c in enumerate(cap):
            if c is not None:
                ret[i], frames[i] = c.read()

        for i, f in enumerate(frames):
            if ret[i] is True:
                # gray[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                # gray[i] = resize_image(gray[i], width=640)
                processed_frames[i] = process(i, f)
                processed_frames[i] = resize_image(processed_frames[i], width=640)
                cv2.imshow(window_titles[i], processed_frames[i])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for c in cap:
        if c is not None:
            c.release()

    cv2.destroyAllWindows()


def main():
    # config_path = "results\\patchcore\\mvtec\\spring_sheet_metal\\run.2024-04-21_09-32-23\\config.yaml"
    model = SpringMetalInspector()
    model._build_model()

    data_paths = get_data_path_generator()

    processed_class = set()

    for side, data_path_generators in data_paths.items():
        # if side != "samping":
        #     continue
        camera_name = f"kamera-{side}"
        for generator in data_path_generators:
            for path in generator:

                if "berr" in str(path) and side == "atas":
                    continue

                if "berr" in str(path) and side == "atas":
                    continue

                # if "dented" not in str(path):
                #     continue

                if "berr" not in str(path):
                    continue

                predict_video(model, path, camera_name)
                # break
            # break


if __name__ == "__main__":
    # main2()
    main()
