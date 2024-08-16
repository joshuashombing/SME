import time
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.post_processing import ImageResult
from omegaconf import OmegaConf
from shapely.geometry import Polygon


def point_within_circle(point, center, radius):
    x, y = point
    center_x, center_y = center
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return distance <= radius


class SpringMetalInspector:
    def __init__(self, config_path: Path | str):
        self.config_path = Path(config_path)
        self.class_names = ["Good", "Defect"]
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    def _build_model(self) -> None:
        config = OmegaConf.load(self.config_path)
        config.visualization.show_images = False
        config.visualization.save_images = False
        config.visualization.log_images = False
        config.visualization.image_save_path = None

        model_filename = "model"
        config.trainer.resume_from_checkpoint = "D:\\maftuh\DATA\\19Apr24-sme-anomalib-train\\20-04-2024-sme-results\\weights\\lightning\\model-atas.ckpt"
        weight_path = config.trainer.resume_from_checkpoint

        self.model = get_model(config)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device)["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # get the transforms
        transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
        image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
        center_crop = config.dataset.get("center_crop")

        if center_crop is not None:
            center_crop = tuple(center_crop)

        normalization = InputNormalizationMethod(config.dataset.normalization)
        self.transform = get_transforms(
            config=transform_config,
            image_size=image_size,
            center_crop=center_crop,
            normalization=normalization
        )
        # self.classifier = SpringClassifier(
        #     model_path="tools/spring_classify.pth",
        #     device=self.device
        # )
        # self.classifier.model.to(self.device)
        # self.segmentor = ChangeOutsideToWhite(
        #     model_path="D:\\Projects\\SME\\anomalib\\tools\\best.pt",
        #     device=self.device
        # )

    @staticmethod
    def visualize(image, batch):
        i = 0
        image_result = ImageResult(
            image=cv2.resize(image, (480, 360)),
            pred_score=batch["pred_scores"][i].cpu().numpy().item(),
            pred_label=batch["pred_labels"][i].cpu().numpy().item(),
            anomaly_map=batch["anomaly_maps"][i].cpu().numpy() if "anomaly_maps" in batch else None,
            pred_mask=batch["pred_masks"][i].squeeze().int().cpu().numpy() if "pred_masks" in batch else None,
            gt_mask=batch["mask"][i].squeeze().int().cpu().numpy() if "mask" in batch else None,
            gt_boxes=batch["boxes"][i].cpu().numpy() if "boxes" in batch else None,
            pred_boxes=batch["pred_boxes"][i].cpu().numpy() if "pred_boxes" in batch else None,
            box_labels=batch["box_labels"][i].cpu().numpy() if "box_labels" in batch else None,
        )
        return image_result

    def predict_defect(self, image: np.ndarray):
        batch = self.transform(image=image)
        batch["image"] = batch["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.model.predict_step(batch, 0)

        batch.update(result)
        image_result = self.visualize(
            image,
            batch
        )
        return image_result

    def remove_background(self, image):
        masks, segmentation = self.segmentor.remove_background(image)

        if masks is None:
            return np.ones_like(image) * 255, False

        polygon = Polygon(segmentation)
        center = polygon.centroid.x, polygon.centroid.y
        image_center = (masks.shape[1] / 2 - 10, masks.shape[0] / 2)
        radius = 50
        # Check if the point is within the circular ROI
        is_within_circle = point_within_circle(center, image_center, radius)

        if not is_within_circle:
            return self.segmentor.return_image(masks, image), False

        return self.segmentor.return_image(masks, image), True

    def draw_grid(self, image, padding_y=-10, circle_radius=100):
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2

        cv2.line(image, (0, center_y + padding_y), (width, center_y + padding_y), (0, 0, 0), 1)
        cv2.line(image, (center_x, 0), (center_x, height), (0, 0, 0), 1)

        cv2.circle(image, (center_x, center_y + padding_y), circle_radius, (0, 0, 0), 1)
        return image

    def predict(self, image: np.ndarray, ori_shape):
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.predict_defect(rgb_img)
        img_show = result.segmentations.copy()
        idx = int(result.pred_label)
        pred_label = self.class_names[idx]
        score = result.pred_score

        img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)
        img_show = cv2.resize(img_show, (ori_shape[1], ori_shape[0]))
        cv2.putText(
            img_show, f"Condition: {pred_label}", (20, 40), cv2.FONT_HERSHEY_DUPLEX,
            0.7, (0, 0, 255) if idx == 1 else (255, 0, 0), 1, cv2.LINE_AA
        )
        cv2.putText(
            img_show, f"Score: {score:.4f}", (20, 60), cv2.FONT_HERSHEY_DUPLEX,
            0.5, (0, 0, 255) if idx == 1 else (255, 0, 0), 1, cv2.LINE_AA
        )
        return img_show

    def _predict(self, image: np.ndarray):
        ori_image = image.copy()
        height, width = image.shape[:2]

        center_x, center_y = width // 2, height // 2
        padding_y = -10
        cv2.line(ori_image, (0, center_y + padding_y), (width, center_y + padding_y), (0, 0, 0), 1)
        cv2.line(ori_image, (center_x, 0), (center_x, height), (0, 0, 0), 1)
        circle_radius = 100
        cv2.circle(ori_image, (center_x, center_y + padding_y), circle_radius, (0, 0, 0), 1)

        # image_copy = image.copy()
        # print(image_copy)
        # is_metal = self.classifier.predict(image)
        # if not is_metal:
        #     print("There is no metal on the image")
        #     return image
        masks, segmentation = self.segmentor.remove_background(image)
        # print(segmentation)
        if masks is None:
            print("There is no metal on the image")

            return np.concatenate([ori_image, np.ones_like(image) * 255], axis=1), 1

        polygon = Polygon(segmentation)
        center = polygon.centroid.x, polygon.centroid.y
        image_center = (masks.shape[1] / 2, masks.shape[0] / 2)
        radius = min(masks.shape[1], masks.shape[0]) / 8
        # Check if the point is within the circular ROI
        is_within_circle = point_within_circle(center, image_center, radius)
        image_segment = self.segmentor.return_image(masks, image)

        if not is_within_circle:
            return np.concatenate([ori_image, image_segment], axis=1), 1

        rgb_img = cv2.cvtColor(image_segment, cv2.COLOR_BGR2RGB)
        result = self.predict_defect(rgb_img)
        img_show = result.segmentations.copy()
        idx = int(result.pred_label)
        pred_label = self.class_names[idx]
        score = result.pred_score

        img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)
        img_show = cv2.resize(img_show, (width, height))
        cv2.putText(
            img_show, f"Condition: {pred_label}", (20, 40), cv2.FONT_HERSHEY_DUPLEX,
            0.7, (0, 0, 255) if idx == 1 else (255, 0, 0), 1, cv2.LINE_AA
        )
        cv2.putText(
            img_show, f"Score: {score:.4f}", (20, 60), cv2.FONT_HERSHEY_DUPLEX,
            0.5, (0, 0, 255) if idx == 1 else (255, 0, 0), 1, cv2.LINE_AA
        )

        img_show = np.concatenate([ori_image, img_show], axis=1)

        return img_show, 2

    def stream(self):
        # self._build_model()
        # Open the default camera (usually the first one)
        cap = cv2.VideoCapture(0)

        # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Turn off autofocus
        # cap.set(cv2.CAP_PROP_EXPOSURE, -7.0)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(f'output_video_{time.time()}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (640*2, 480))

        # Check if the camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        predicting = False

        # Keep reading frames from the camera and displaying them
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # If frame is read correctly, ret is True
            if not ret:
                print("Error: Could not read frame.")
                break

            ori_image = frame.copy()
            # ori_image = self.draw_grid(ori_image)

            new_frame, is_object = self.remove_background(image=frame.copy())
            if is_object:
                new_frame = self.predict(new_frame, ori_image.shape)

            frame_show = np.concatenate([ori_image, new_frame], axis=1)
            cv2.imshow('Camera', frame_show)
            # out.write(frame_show)

            # if cv2.waitKey(1) & 0xFF == ord('p'):
            #     predicting = True
            #
            # if cv2.waitKey(1) & 0xFF == ord('o'):
            #     predicting = False

            # Check if the user pressed the 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # elif cv2.waitKey(1) & 0xFF == ord('p'):
            #     predicting = True
            # elif cv2.waitKey(1) & 0xFF == ord('o'):
            #     predicting = False

        # Release the camera and close the OpenCV windows
        cap.release()
        # out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config_path = "D:\\maftuh\\DATA\\19Apr24-sme-anomalib-train\\20-04-2024-sme-results\\config.yaml"
    predictor = SpringMetalInspector(config_path)
    predictor._build_model()
    predictor.stream()

    # predictor._build_model()
    # img_path = "D:\\Projects\\patchcore-few-shot\\datasets\\mvtec\\spring_sheet_metal\\test\\good\\ok (2).jpg"
    # img = cv2.imread(img_path)
    # res = predictor.predict(img)
    # print(res)
    # img_show = predictor.visualize(img, res)
    # print(img_show)
    # print(res)
