

DETECTOR_PARAMS_TOP_CAMERA = dict(
    path="runs/segment/train2/weights/best.pt",
    camera_name="kamera-atas",
    output_patch_shape=(680, 560),
    distance_thresholds=(0.4, 0.5)
)

DETECTOR_PARAMS_SIDE_CAMERA = dict(
    path="runs/detect/train2/weights/best.pt",
    camera_name="kamera-samping",
    output_patch_shape=(680, 320),
    distance_thresholds=(0.4, 0.5)
)

PREPROCESSOR_PARAMS_TOP_CAMERA = dict(
    target_shape=(680, 560),
    distance_thresholds=(0.4, 0.5),
    camera_name="kamera-atas"
)

PREPROCESSOR_PARAMS_SIDE_CAMERA = dict(
    target_shape=(680, 320),
    distance_thresholds=(0.4, 0.5),
    camera_name="kamera-samping"
)

INSPECTOR_PARAMS_TOP_CAMERA = dict(
    config_path="results/patchcore/mvtec/spring_sheet_metal/run.2024-05-18_23-18-30/config.yaml",
    root=None,
)

INSPECTOR_PARAMS_SIDE_CAMERA = dict(
    config_path="results/patchcore/mvtec/spring_sheet_metal/run.2024-05-18_23-46-04/config.yaml",
    root=None
)


