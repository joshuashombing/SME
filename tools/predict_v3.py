import time
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2

from anomalib.models.detection.defect_detection_v1 import DefectPredictorThread, DefectPreprocessor
from anomalib.models.detection.spring_metal_detection import SpringMetalDetector
from anomalib.pre_processing.prepare_defect import create_anomalib_data
from anomalib.pre_processing.utils import resize_image, get_now_str
from fps import FPS
from config import DETECTOR_PARAMS_TOP_CAMERA, PREPROCESSOR_PARAMS_TOP_CAMERA, DETECTOR_PARAMS_SIDE_CAMERA, \
    PREPROCESSOR_PARAMS_SIDE_CAMERA


def get_data_path_generator():
    data_root1 = Path("datasets")
    # data_root2 = Path("D:\\maftuh\\DATA\\MV-CAMERA\\MV-CE120-10GM (00DA2265775)")
    # data_root3 = Path("D:\\maftuh\\DATA\\MV-CAMERA\\MV-CE120-10GM (00DA2265758)")
    # data_root4 = Path("D:\\maftuh\\DATA\\16Apr24-sme-spring sheet")

    data_paths = {
        "atas": [
            data_root1.glob("**/atas/*.avi"),
            # data_root2.glob("**/*.avi"),
            # data_root4.glob("**/*atas*.avi"),
        ],
        # "samping": [
        #     data_root1.glob("**/samping/*.avi"),
        # data_root3.glob("**/*.avi"),
        # data_root4.glob("**/*samping*.avi"),
        # ]
    }
    return data_paths


def send_signal(result_queue: Queue):
    while True:
        if not result_queue.empty():
            result = result_queue.get()
            print(result)


def predict_video(
        detector: SpringMetalDetector,
        preprocessor: DefectPreprocessor,
        video_path: Path,
        class_name: str,
        output_dir: Path
):
    cap = cv2.VideoCapture(str(video_path))
    print(f"Processing {video_path}")
    fps = FPS().start()

    class_names = {
        0: "good",
        1: "defect"
    }
    color_map = {
        0: (0, 255, 0),
        1: (0, 0, 255)
    }
    # detector.tracker.frame_

    # detector.start()
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            cap.release()
            break

        # time.sleep(0.05)

        result = detector.predict(frame)
        result = detector.track(result)

        patches_map = preprocessor.object_post_process(frame, result)

        for position, object_id, patch in patches_map:
            create_anomalib_data(
                cv2.cvtColor(patch, cv2.COLOR_BGR2RGB),
                output_dir,
                class_name=class_name,
                filename=f"{video_path.stem}_object_{object_id}.png",
                create_mask=class_name != "good"
            )

        # objects_copy = preprocessor.patches.copy()
        #
        # for object_id, all_exists in objects_copy.items():
        #     all_exists = [patch is not None for patch in patches]
        #     if all(all_exists):
        #         patches = detector.patches.pop(object_id)
        #         # frame_queue.put(patches)
        #         print(len(patches))
        #         # predictor.process(patches)
        #
        #         # result_defect = predictor.get_result()
        #         # if result_defect is not None:
        #         #     scores, labels = result_defect
        #         #     scores = result_defect["pred_scores"].cpu().numpy()
        #         #     labels = result_defect["pred_labels"].cpu().long().numpy()
        #         #     print(scores)
        #         #     print(labels)
        #         for i in range(len(patches)):
        #             create_anomalib_data(
        #                 cv2.cvtColor(patches[i], cv2.COLOR_BGR2RGB),
        #                 output_dir,
        #                 class_name=class_name,
        #                 filename=f"{video_path.stem}_object_{object_id}_{i}.png",
        #                 create_mask=class_name != "good"
        #             )

        # print()

        # if len(objects) > 0:
        #     print(objects)

        # if len(result) > 1:
        #     patches = detector.post_multiprocess(frame, result)
        # else:
        # patches = detector.post_process(frame, result)
        frame = result.plot()

        # detector.process(frame)
        # out = detector.get_result()
        # frame = resize_image(frame, width=640)
        for object_id, pt in detector.tracker.objects.items():
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
            cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 255, 0), 2)

        # if out is not None:
        #     result, patches = out
        #     boxes = result.boxes.xyxyn
        #     scores = [None for _ in range(len(boxes))]
        #     labels = [None for _ in range(len(boxes))]
        #     if len(patches) > 0:
        #         predictor.process(patches)
        #         result_defect = predictor.get_result()
        #         if result_defect is not None:
        #             scores = result_defect["pred_scores"].cpu().numpy()
        #             labels = result_defect["pred_labels"].cpu().long().numpy()
        #
        #     for box, score, label in zip(boxes, scores, labels):
        #         draw_bbox(
        #             frame,
        #             box=box,
        #             label=class_names[label] if label is not None else None,
        #             score=score,
        #             color=color_map[label] if label is not None else (255, 255, 255)
        #         )

        cv2.imshow(video_path.name, frame)
        fps.update()

        # cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if fps.elapsed() >= 1:
            # print(fps.fps())
            fps.start()

    # detector.stop()
    cap.release()
    cv2.destroyAllWindows()


def main():
    data_paths = get_data_path_generator()

    for side, data_path_generators in data_paths.items():
        for generator in data_path_generators:
            for path in generator:
                predict_video(path)
                break


def main2():
    # data_root = Path("D:\\maftuh\\Projects\\SME\\anomalib\\datasets").glob("**/*.avi")
    output_root_dir = Path(f"D:\\maftuh\\DATA\\DefectDetectionDataset\\{get_now_str()}")

    detector_params = {
        "atas": DETECTOR_PARAMS_TOP_CAMERA,
        "samping": DETECTOR_PARAMS_SIDE_CAMERA
    }
    preprocessor_params = {
        "atas": PREPROCESSOR_PARAMS_TOP_CAMERA,
        "samping": PREPROCESSOR_PARAMS_SIDE_CAMERA
    }

    video_paths = [
        *[path for path in Path("D:\\maftuh\\DATA\\2024-05-15_13-00").glob("**/*.avi")],
        *[path for path in Path("D:\\maftuh\\DATA\\datasets-130524").glob("**/*.avi")]
    ]

    for video_path in video_paths:
        print(video_path)
        camera_side, class_name, _ = video_path.stem.split("_")
        side = camera_side.split("-")[-1]

        if side not in {"atas", "samping"}:
            continue

        if side == "atas":
            continue

        output_dir = output_root_dir / f"{side}/spring_sheet_metal"
        output_dir.mkdir(parents=True, exist_ok=True)

        detector = SpringMetalDetector(**detector_params[side]).build()
        preprocessor = DefectPreprocessor(**preprocessor_params[side])

        if side == "atas":
            if class_name == "berr":
                class_name = "good"
        else:
            if class_name == "dented":
                continue

        predict_video(detector, preprocessor, video_path, class_name, output_dir=output_dir)

    # predictor.stop()


if __name__ == "__main__":
    # detector = SpringMetalDetector(
    #     path="runs/segment/train/weights/best.pt",
    #     pre_processor=lambda x: resize_image(x, width=640),
    #     output_patch_shape=(680, 560),
    #     camera_name="kamera-atas",
    # )
    # detector = SpringMetalDetector(
    #     path="tools\\yolov8.pt",
    #     transform=lambda x: resize_image(x, width=640),
    #     output_patch_shape=(680, 320),
    #     camera_name="kamera-samping"
    # )
    # detector.build()
    # preprocessor = DefectPreprocessor()
    # predictor = DefectPredictorThread(
    #     config_path="results/patchcore/mvtec/spring_sheet_metal/run.2024-05-15_01-26-33/config.yaml",
    #     num_workers=8
    # )

    # frame_queue = Queue()
    # result_queue = Queue()
    # predictor.start(frame_queue, result_queue)

    # show_thread = Thread(target=send_signal, args=(result_queue,))
    # show_thread.daemon = True
    # show_thread.start()
    #
    main2()

    # predictor.stop()
    # show_thread.start()
