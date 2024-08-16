
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent


DEBUG = True


RESULT_DIR = Path(r"D:\InspectionResult\SpringSheetMetal")
SAVE_IMAGE = True
SAVE_MASK = False
# PUSH_OBJECT = True
SAVE_VIDEO = False

NUM_INSPECTION_PROCESS = 3
TIME_TO_PUSH_OBJECT = 2.5  # time to push object after disappear in second
DELAY_SYSTEM = 0.05
WARMING_TIMES_MODEL = 3

JUDGMENT_METHOD = "any"  # {voting, average, any, at_least_two}
IMAGE_THRESHOLD = None
# IMAGE_THRESHOLD = 47.11189651489258


DETECTOR_PARAMS_TOP_CAMERA = dict(
    path=BASE_DIR / "runs/segment/train2/weights/best.pt",
    camera_name="kamera-atas"
)

DETECTOR_PARAMS_SIDE_CAMERA = dict(
    path=BASE_DIR / "runs/detect/train2/weights/best.pt",
    camera_name="kamera-samping"
)

PREPROCESSOR_PARAMS_TOP_CAMERA = dict(
    target_shape=(680, 480),
    distance_thresholds=(0.4, 0.5),
    camera_name="kamera-atas"
)

PREPROCESSOR_PARAMS_SIDE_CAMERA = dict(
    target_shape=(680, 240),
    distance_thresholds=(0.4, 0.5),
    camera_name="kamera-samping"
)

# dev
# INSPECTOR_PARAMS_TOP_CAMERA = dict(
#     config_path=BASE_DIR / "results/kamera-atas/run.2024-06-30_11-57-26/config.yaml",
#     root=BASE_DIR,
# )
#
# INSPECTOR_PARAMS_SIDE_CAMERA = dict(
#     config_path=BASE_DIR / "results/patchcore/mvtec/kamera-samping/run.2024-06-04_23-33-13/config.yaml",
#     root=BASE_DIR
# )

# prod
INSPECTOR_PARAMS_TOP_CAMERA = dict(
    config_path=BASE_DIR / "results/kamera-atas/run.2024-07-14_23-20-38/config.yaml",
    use_openvino=True,
    # root=BASE_DIR,
)

INSPECTOR_PARAMS_SIDE_CAMERA = dict(
    config_path=BASE_DIR / "results/patchcore/mvtec/kamera-samping/run.2024-06-13_13-48-03/config.yaml",
    # root=BASE_DIR
)



