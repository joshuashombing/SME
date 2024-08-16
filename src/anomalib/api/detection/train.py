# yolo segment train data=src/anomalib/api/detection/springsh.yaml model=yolov8n-seg.pt epochs=100 imgsz=640 batch=32
# yolo segment train data=src/anomalib/api/detection/springsh.yaml model=runs/segment/train/weights/best.pt epochs=50 imgsz=640 batch=32 single_cls=True overlap_mask=False agnostic_nms=True nms=True

# yolo detect train data=src/anomalib/api/detection/springsh.yaml model=tools/yolov8.pt epochs=20 imgsz=640 batch=32 single_cls=True overlap_mask=False agnostic_nms=True nms=True
# yolo detect train data=./springsh.yaml model=./yolov8.pt epochs=50 imgsz=640 batch=64 single_cls=True overlap_mask=False agnostic_nms=True nms=True
