import time
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_video(model: SpringMetalDetector, class_id, video_path: Path, output_dir: Path = None):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(fps / 2))
    frame_count = 0

    base_filename = f"{video_path.stem}_frame"
    print(f"Processing {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            cap.release()
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        ori_frame = frame.copy()
        results = model.detect(frame)

        filename = f"{base_filename}_{str(time.time()).replace('.', '_')}.jpg"
        bbox_filename = f"{Path(filename).stem}_bbox.jpg"
        label_filename = f"{Path(filename).stem}.txt"

        for result in results:
            frame = draw_bbox(frame, result)
            _, _, xyxyn = result
            with open(output_dir / label_filename, "a") as f:
                x1, y1, x2, y2 = xyxyn
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                f.write(f"{class_id} {x} {y} {w} {h}" + "\n")

        frame = resize_image(frame, width=640)
        ori_frame = resize_image(ori_frame, width=640)

        if len(results) > 0:
            cv2.imwrite(str(output_dir / filename), ori_frame)
            cv2.imwrite(str(output_dir / bbox_filename), frame)

        # cv2.imshow(video_path.name, frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    # cv2.destroyAllWindows()


def transform_bbox(result, img_shape):
    height, width = img_shape[:2]

    cls, conf, xyxyn = result
    x1, y1, x2, y2 = xyxyn

    x1 = int(x1 * width)
    y1 = int(y1 * height)
    x2 = int(x2 * width)
    y2 = int(y2 * height)
    return x1, y1, x2, y2


def is_valid_bbox(bbox_n, aspect_ratio=1.3, aspect_ratio_tol=0.2, min_area=0.01):
    x1, y1, x2, y2 = bbox_n
    wn = abs(x2 - x1)
    hn = abs(y2 - y1)
    area_n = wn * hn
    ratio = wn / hn
    print(area_n, ratio)
    return area_n >= min_area and abs(ratio - aspect_ratio) < aspect_ratio_tol


def extract_video_anomalib(model: SpringMetalDetector, video_path: Path, output_dir: Path = None,
                           filter_bbox_param=None):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(fps / 2))
    frame_count = 0

    base_filename = f"{video_path.stem}_frame"
    print(f"Processing {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            cap.release()
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        results = model.detect(frame)

        for i, result in enumerate(results):
            x1, y1, x2, y2 = transform_bbox(result, frame.shape)

            if filter_bbox_param is not None:
                if not is_valid_bbox(result[-1], **filter_bbox_param):
                    continue

            cropped_frame = frame[y1:y2, x1:x2]

            filename = f"{base_filename}_{str(time.time()).replace('.', '_')}_{i}.jpg"
            cv2.imwrite(str(output_dir / filename), cropped_frame)


def get_data_path_generator():
    data_root1 = Path("D:\\maftuh\\DATA\\SME-VIDEO-Dataset")
    data_root2 = Path("D:\\maftuh\\DATA\\MV-CAMERA\\MV-CE120-10GM (00DA2265775)")
    data_root3 = Path("D:\\maftuh\\DATA\\MV-CAMERA\\MV-CE120-10GM (00DA2265758)")
    data_root4 = Path("D:\\maftuh\\DATA\\16Apr24-sme-spring sheet")

    data_paths = {
        "atas": [
            data_root1.glob("**/atas/*.avi"),
            data_root2.glob("**/*.avi"),
            data_root4.glob("**/*atas*.avi"),
        ],
        "samping": [
            data_root1.glob("**/samping/*.avi"),
            data_root3.glob("**/*.avi"),
            data_root4.glob("**/*samping*.avi"),
        ]
    }
    return data_paths


def main():
    data_paths = get_data_path_generator()
    output_dir = Path("D:\\maftuh\\DATA\\ObjectDetectionDataset20042024")

    class_map = {
        "atas": 0,
        "samping": 1
    }

    model = SpringMetalDetector(max_num_detection=20, distance_thresholds=[0.3, 0.5])

    for class_name, data_path_generators in data_paths.items():
        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        print(class_map[class_name], class_name)
        for generator in data_path_generators:
            for path in generator:
                extract_video(model, class_map[class_name], path, output_dir=output_class_dir)


def main2():
    data_root1 = Path("D:\\maftuh\\DATA\\SME-VIDEO-Dataset")
    data_root2 = Path("D:\\maftuh\\DATA\\16Apr24-sme-spring sheet")
    output_dir = Path("D:\\maftuh\\DATA\\DefectDetectionDataset21042024")

    data_dir_map = {}

    for path in tqdm(data_root1.glob('**/*.avi')):
        model_name = path.parents[0].name.lower()
        class_name = path.parents[1].name.lower()

        if model_name not in data_dir_map:
            data_dir_map[model_name] = {}

        if class_name not in data_dir_map[model_name]:
            data_dir_map[model_name][class_name] = []

        data_dir_map[model_name][class_name].append(path)

    for path in tqdm(data_root2.glob('**/*.avi')):
        class_name, model_name = path.name.split("-")
        class_name, model_name = class_name.strip(), Path(model_name.split()[1].strip()).stem

        if class_name == "double stamp":
            class_name = "shift"

        if model_name not in data_dir_map:
            data_dir_map[model_name] = {}

        if class_name not in data_dir_map[model_name]:
            data_dir_map[model_name][class_name] = []

        data_dir_map[model_name][class_name].append(path)

    params = {
        "atas": {
            "min_area": 0.01,
            "aspect_ratio": 1,
            "aspect_ratio_tol": 0.3,
        },
        "samping": {
            "min_area": 0.005,
            "aspect_ratio": 2.3,
            "aspect_ratio_tol": 0.3,
        }
    }

    model = SpringMetalDetector(max_num_detection=20, distance_thresholds=[0.5, 0.5])
    for model_name in data_dir_map:
        for class_name in data_dir_map[model_name]:
            output_class_dir = output_dir / model_name / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            for path in data_dir_map[model_name][class_name]:
                extract_video_anomalib(model, path, output_dir=output_class_dir)


if __name__ == "__main__":
    main()
    # print(get_data_path_generator_v2())
    # main2()
