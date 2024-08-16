from collections import OrderedDict
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.transform import rotate

from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.pre_processing.transforms import Compose
from anomalib.pre_processing.utils import add_background
from .utils import transform_mask_to_rect, get_rect_angle_rotation, expand_bbox, unormalize_bbox, select_closest_bbox, \
    center_crop


class DefectResult:
    def __init__(self, result):
        self.scores = result["pred_scores"].cpu().numpy()
        self.labels = result["pred_labels"].cpu().numpy()
        self.anomaly_maps = result["anomaly_maps"].cpu().numpy()
        del result


class DefectPredictor:
    def __init__(
            self,
            config_path,
            device="auto",
            num_workers=4,
            root=None,
            use_openvino=False
    ):
        self.config_path = config_path
        self.num_workers = num_workers
        self.root = Path("." if root is None else root)

        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.default_transform = None
        self.threshold = None
        self.pixel_threshold = None
        self.use_openvino = use_openvino
        self.predict = self.predict_v1 if self.use_openvino else self.predict_v0
        self.build = self.build_v1 if self.use_openvino else self.build_v0

    # def build(self):
    #     if self.use_openvino:
    #         return self.build_v1()
    #     return self.build_v0()

    def build_v1(self):
        from anomalib.deploy import OpenVINOInferencer
        # config = OmegaConf.load(self.config_path)
        model_path = Path(self.config_path).parent / "weights/openvino/model.bin"
        metadata_path = Path(self.config_path).parent / "weights/openvino/metadata.json"
        self.model = OpenVINOInferencer(
            path=model_path, metadata=metadata_path, device="GPU"
        )
        return self

    def build_v0(self):
        config = OmegaConf.load(self.config_path)
        config.visualization.show_images = False
        config.visualization.save_images = False
        config.visualization.log_images = False
        config.visualization.image_save_path = None

        config.trainer.resume_from_checkpoint = str(Path(self.config_path).parent / "weights/lightning/model.ckpt")
        weight_path = config.trainer.resume_from_checkpoint

        self.model = get_model(config)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device)["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.threshold = self.model.image_threshold.value.cpu().numpy()
        self.pixel_threshold = self.model.pixel_threshold.value

        transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
        image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
        center_crop_ = config.dataset.get("center_crop")

        if center_crop_ is not None:
            center_crop_ = tuple(center_crop_)

        normalization = InputNormalizationMethod(config.dataset.normalization)
        self.default_transform = get_transforms(
            config=transform_config,
            image_size=image_size,
            center_crop=center_crop_,
            normalization=normalization
        )
        # path = self.root / "sample/images/dented/kamera-atas_dented_1714378296-9803188_object_21_0.png"
        # for _ in range(5):
        #     self.predict([read_image(path)])
        return self

    def _pre_process_single_image(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch = self.default_transform(image=image)
        batch["image"] = batch["image"].unsqueeze(0)
        return batch

    # @timeit
    def _pre_process_batch(self, images):
        if len(images) == 1:
            return self._pre_process_single_image(images[0])

        collected_batch = None
        for image in images:
            batch = self._pre_process_single_image(image)
            if collected_batch is None:
                collected_batch = batch.copy()
            else:
                collected_batch["image"] = torch.cat((collected_batch["image"], batch["image"]), dim=0)

        return collected_batch

    def pre_process(self, images):
        return self._pre_process_batch(images)

    # def predict(self, images):
    #     return self.predict_v0(images)

    # @timeit
    def predict_v0(self, images) -> dict:
        if len(images) == 0:
            return {}

        batch = self.pre_process(images)
        with torch.no_grad():
            batch["image"] = batch["image"].to(self.device)
            results = self.model.predict_step(batch, 0)

        results = DefectResult(results)
        return results.__dict__

    def predict_v1(self, images) -> dict:
        if len(images) == 0:
            return {}

        # for image in images:
        predictions = self.model.predict(image=images[0])

        return {
            "scores": [predictions.pred_score],
            "labels": [predictions.pred_label],
            "anomaly_maps": [predictions.anomaly_map],
            "heatmaps": [predictions.heat_map]
        }


class DefectPreprocessor:
    def __init__(
            self, distance_thresholds=(0.4, 0.5), conf_threshold=0.5, target_shape=(680, 560),
            expand_bbox_percentage=0.2, camera_name="kamera-atas"
    ):
        self.distance_thresholds = distance_thresholds
        self.conf_threshold = conf_threshold
        self.target_shape = target_shape
        self.expand_bbox_percentage = expand_bbox_percentage
        self.camera_name = camera_name

        self.patches = OrderedDict()

    def _post_process_atas(self, image, result, **kwargs):
        if len(result) == 0:
            return []

        target_shape = kwargs.get("target_shape", self.target_shape)
        distance_thresholds = kwargs.get("distance_thresholds", self.distance_thresholds)
        conf_threshold = kwargs.get("conf_threshold", self.conf_threshold)
        expand_bbox_percentage = kwargs.get("expand_bbox_percentage", self.expand_bbox_percentage)

        get_angle = Compose([
            transform_mask_to_rect,
            get_rect_angle_rotation
        ])
        transform_bbox = Compose([
            lambda xyxyn: expand_bbox(xyxyn, percentage=expand_bbox_percentage),
            lambda xyxyn: unormalize_bbox(xyxyn, src_shape=image.shape, transcend=False)
        ])

        result_filter = select_closest_bbox(
            result,
            distance_thresholds=distance_thresholds,
            conf_threshold=conf_threshold
        )

        patches = []
        for mask, box in zip(result_filter.masks.xy, result_filter.boxes.xyxyn.numpy()):
            x1, y1, x2, y2 = transform_bbox(box)
            patch = image[y1:y2, x1:x2]
            angle = get_angle(mask)
            rotated = rotate(patch, angle, resize=False, preserve_range=True).astype(np.uint8)
            cropped = center_crop(rotated, new_width=target_shape[0], new_height=target_shape[1])
            patches.append(cropped)
        return patches

    # @timeit
    def _post_process_samping(self, image, result, **kwargs):
        expand_bbox_percentage = kwargs.get("expand_bbox_percentage", self.expand_bbox_percentage)
        target_shape = kwargs.get("target_shape", self.target_shape)
        distance_thresholds = kwargs.get("distance_thresholds", self.distance_thresholds)
        conf_threshold = kwargs.get("conf_threshold", self.conf_threshold)

        transform_bbox = Compose([
            lambda xyxyn: expand_bbox(xyxyn, percentage=expand_bbox_percentage),
            lambda xyxyn: unormalize_bbox(xyxyn, src_shape=image.shape, transcend=False)
        ])

        result_filter = select_closest_bbox(
            result,
            distance_thresholds=distance_thresholds,
            conf_threshold=conf_threshold
        )
        patches = []
        for box in result_filter.boxes.xyxyn.numpy():
            x1, y1, x2, y2 = transform_bbox(box)
            cropped = image[y1:y2, x1:x2]
            cropped = add_background(cropped, target_shape)
            patches.append(cropped)
        return patches

    def post_process(self, image, result, **kwargs):
        camera_name = kwargs.get("camera_name", self.camera_name)

        if "camera_name" in kwargs:
            del kwargs["camera_name"]

        if camera_name == "kamera-atas":
            return self._post_process_atas(image, result, **kwargs)

        return self._post_process_samping(image, result, **kwargs)

    def object_post_process(self, image, result, **kwargs):
        # if not hasattr(result.boxes, "track_id"):
        #     raise ValueError(
        #         "The attribute 'track_id' is missing from 'result.boxes'. "
        #         "Please ensure 'result.boxes' includes 'track_id'."
        #     )

        camera_name = kwargs.get("camera_name", self.camera_name)
        target_shape = kwargs.get("target_shape", self.target_shape)

        get_angle = Compose([
            transform_mask_to_rect,
            get_rect_angle_rotation
        ])
        transform_bbox = Compose([
            lambda xyxyn: expand_bbox(xyxyn, percentage=self.expand_bbox_percentage),
            lambda xyxyn: unormalize_bbox(xyxyn, src_shape=image.shape, transcend=False)
        ])

        lines = [0.65, 0.5, 0.35]

        current_patches = []
        for i, (box, object_id) in enumerate(zip(result["boxes_n"], result["track_ids"])):

            # if object_id not in self.patches:
            #     self.patches[object_id] = [False for _ in range(len(lines))]
            #
            center_x = (box[0] + box[2]) / 2
            #
            # for j, line in enumerate(lines):
            #     distance = abs(line - center_x)
            #
            #     if distance > 0.07 or self.patches[object_id][j]:
            #         continue

            x1, y1, x2, y2 = transform_bbox(box)
            cropped = image[y1:y2, x1:x2]
            if camera_name == "kamera-atas" and result["masks"] is not None:
                mask = result["masks"][i]
                angle = get_angle(mask)
                h, w = cropped.shape[:2]
                matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                cropped = cv2.warpAffine(cropped, matrix, (w, h))
                cropped = center_crop(cropped, new_width=target_shape[0], new_height=target_shape[1])
            else:
                cropped = add_background(cropped, target_shape)

            current_patches.append((center_x, object_id, cropped))
                # self.patches[object_id][j] = True

        return current_patches
