from datetime import datetime
from pathlib import Path

import cv2
from anomalib.pre_processing.prepare_yolo import create_yolo_data, clear_missing, copy_data, create_yolo_data_2, \
    create_yolo_data_class_pcb_counter
from anomalib.pre_processing.utils import validate_path, get_now_str
from sklearn.model_selection import train_test_split


def prepare_yolo_from_video(
        video_path, output_dir, fps=2, class_id=0, filename=None, width=640, height=None,
        use_yolo=False, ckpt_path=None, use_class_pcb_counter=False
):
    print(f"Processing {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / fps))
    frame_count = 0

    if not cap.isOpened():
        print("Error: Could not open video.", video_path)
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        if use_yolo:
            create_yolo_data_2(frame, output_dir, class_id=class_id, filename=filename, width=width, height=height,
                               ckpt_path=ckpt_path, device="cpu")
        elif use_class_pcb_counter:
            create_yolo_data_class_pcb_counter(frame, output_dir, class_id=class_id, filename=filename, width=width,
                                               height=height)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            create_yolo_data(frame, output_dir, class_id=class_id, filename=filename, width=width, height=height)
    cap.release()


def split_data(data_dir, test_size=0.3, random_state=42):
    class_range = [
        ("2024-04-28_01-06-04", "2024-04-28_01-20-02"),
        ("2024-04-28_01-20-04", "2024-04-28_01-21-17"),
        ("2024-04-28_01-21-29", "2024-04-28_01-22-17"),
        ("2024-04-28_01-22-26", "2024-04-28_01-22-34"),
        ("2024-04-28_01-22-36", "2024-04-28_01-25-04"),
        ("2024-04-28_01-25-06", "2024-04-28_01-26-56")
    ]

    data_dir = validate_path(data_dir)
    images_dir = data_dir / "images"

    X = []
    y = []
    for file_path in images_dir.iterdir():
        file_id = file_path.stem
        file_id = "-".join(file_id.split("-")[:-1])

        # Convert file id to datetime
        file_datetime = datetime.strptime(file_id, "%Y-%m-%d_%H-%M-%S")

        class_id = 0
        # Determine which class range the file falls into
        for i, (start, end) in enumerate(class_range):
            start_datetime = datetime.strptime(start, "%Y-%m-%d_%H-%M-%S")
            end_datetime = datetime.strptime(end, "%Y-%m-%d_%H-%M-%S")
            if start_datetime <= file_datetime <= end_datetime:
                class_id = i
                break

        X.append(file_path.name)
        y.append(class_id)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    copy_data(data_dir, X_train, split="train")
    copy_data(data_dir, X_test, split="val")


def split_data_v2(root_dir, test_size=0.3, random_state=42, dest_dir=None):
    root_dir = validate_path(root_dir)

    for i, data_dir in enumerate(root_dir.iterdir()):
        if "split" in data_dir.name or not data_dir.is_dir():
            continue
        images_dir = data_dir / "images"
        X = [file_path.name for file_path in images_dir.iterdir()]
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        copy_data(data_dir, X_train, split="train", dest_dir=dest_dir)
        copy_data(data_dir, X_test, split="val", dest_dir=dest_dir)


def get_data_path_generator():
    data_root1 = Path("D:\\maftuh\\DATA\\SME-VIDEO-Dataset")
    data_root2 = Path("D:\\maftuh\\DATA\\MV-CAMERA\\MV-CE120-10GM (00DA2265775)")
    data_root3 = Path("D:\\maftuh\\DATA\\MV-CAMERA\\MV-CE120-10GM (00DA2265758)")
    data_root4 = Path("D:\\maftuh\\DATA\\16Apr24-sme-spring sheet")
    data_root5 = Path("D:\\maftuh\\Projects\\SME\\anomalib\\datasets")
    data_root6 = Path("D:\\maftuh\\DATA\\2024-05-15_13-00")

    data_paths = {
        # "atas": [
        #     data_root1.glob("**/atas/*.avi"),
        #     data_root2.glob("**/*.avi"),
        #     data_root4.glob("**/*atas*.avi"),
        #     data_root5.glob("**/*atas*.avi"),
        #     data_root6.glob("**/*atas*.avi"),
        # ],
        "samping": [
            # data_root1.glob("**/samping/*.avi"),
            # data_root3.glob("**/*.avi"),
            # data_root4.glob("**/*samping*.avi"),
            data_root5.glob("**/*samping*.avi"),
            data_root6.glob("**/*samping*.avi"),
        ]
    }
    class_map = {
        "atas": 0,
        "samping": 1
    }
    return data_paths, class_map


def main():
    data_paths, class_map = get_data_path_generator()
    output_dir = Path("D:\\maftuh\\DATA\\ObjectSegmentationDataset\\atas")

    for class_name, data_path_generators in data_paths.items():
        for generator in data_path_generators:
            for path in generator:
                prepare_yolo_from_video(
                    path,
                    output_dir,
                    class_id=class_map[class_name],
                    fps=2
                )
                # break
            # break


def main2():
    ckpt_paths = {
        "atas": "runs/segment/train/weights/best.pt",
        "samping": "tools/yolov8.pt"
    }
    data_paths, class_map = get_data_path_generator()
    output_dir = Path("D:\\maftuh\\DATA\\YoloTrainDataset") / get_now_str()

    for side, data_path_generators in data_paths.items():
        for i, generator in enumerate(data_path_generators):
            for video_path in generator:
                prepare_yolo_from_video(
                    video_path,
                    output_dir / side / f"video_{i}",
                    class_id=class_map[side],
                    fps=2,
                    use_yolo=True,
                    ckpt_path=ckpt_paths[side],
                    filename=f"{video_path.stem}.jpg"
                )


def main3():
    input_dir = Path("D:\\maftuh\\DATA\\ObjectSegmentationDataset\\atas")
    output_dir = Path("D:\\maftuh\\DATA\\ObjectSegmentationDataset\\atas")

    prepare_yolo_from_video(
        input_dir,
        output_dir,
        class_id=0,
        fps=2,
        use_class_pcb_counter=True
    )


def main4():
    output_dir = Path("C:\\Users\\User\\Downloads\\output-smt-pcb-counter") / get_now_str()

    data_dir = Path("C:\\Users\\User\\Documents\\Charlie\\ws")
    for video_path in data_dir.glob("**/*.mp4"):
        prepare_yolo_from_video(
            video_path,
            output_dir,
            class_id=0,
            fps=2,
            use_yolo=True,
            ckpt_path="C:\\Users\\User\\Downloads\\smt-pcb-counter-detection-train\\train2\\weights\\best.pt",
            filename=f"{video_path.stem}.jpg"
        )
        break


def clear_images():
    data_dir = Path("D:\\maftuh\\DATA\\YoloTrainDataset\\2024-05-17_00-45-40-487228\\samping")
    for video_dir in data_dir.iterdir():
        if not video_dir.is_dir():
            continue
        clear_missing(video_dir)


if __name__ == "__main__":
    main4()
    # clear_images()
    # split_data_v2(
    #     root_dir="D:\\maftuh\\DATA\\YoloTrainDataset\\2024-05-17_00-45-40-487228\\samping",
    #     dest_dir="D:\\maftuh\\DATA\\YoloTrainDataset\\2024-05-17_00-45-40-487228\\samping_split"
    # )
