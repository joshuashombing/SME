import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from anomalib.models.detection.metal_detector import detect_metal, detect_metal_raw
from anomalib.models.detection.spring_metal_detection import SpringMetalDetector
from anomalib.models.detection.utils import draw_bbox, resize_image, expand_bbox, make_bbox_square, unormalize_bbox
from deskew import determine_skew
from scipy import stats as st
from skimage.transform import rotate


def metal_detector(image, aspect_ratio=1.3, aspect_ratio_tol=0.2, min_area=0.01):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (13, 13), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7)

    # Morph close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours, sort for largest contour, draw contour
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    return image, thresh, cnts

def segment_image_by_color(img):
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    HSV_mask = cv2.inRange(img_HSV, (0, 0, 20), (179, 255, 255))
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    YCrCb_mask = cv2.inRange(img_YCrCb, (18, 0, 0), (255, 255, 255))
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
    global_mask = cv2.medianBlur(global_mask, 3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
    segmented_image = cv2.bitwise_and(img, img, mask=global_mask)
    
    return segmented_image, global_mask


def segment_mean_shift(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    result_bboxes = []
    for cnt in contours:
        x1, y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        # x, y, w, h = cv2.boundingRect(approx)
        # wn, hn = w / image.shape[1], h / image.shape[0]
        # ratio = float(w) / h
        # area = wn * hn
        #
        # if area < 0.01:
        #     continue

        if len(approx) <= 6:
            x, y, w, h = cv2.boundingRect(cnt)

            wn, hn = w / image.shape[1], h / image.shape[0]
            ratio = float(w) / h
            area = wn * hn

            if area < 0.01:
                continue

            result_bboxes.append([x, y, x + w, y + h])

            if 0.9 <= ratio <= 1.1:
                result = cv2.drawContours(result, [cnt], -1, (0, 255, 255), 3)
                cv2.putText(result, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(result, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                result = cv2.drawContours(result, [cnt], -1, (0, 255, 0), 3)

    return result, result_bboxes


def extract_video(model: SpringMetalDetector, class_id, video_path: Path, output_dir: Path = None):
    cap = cv2.VideoCapture(str(video_path))

    base_filename = f"{video_path.stem}_frame"
    print(f"Processing {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            cap.release()
            break

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

        cv2.imshow(video_path.name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def _create_mask(img, fill=255):
    assert img is not None, "file could not be read, check with os.path.exists()"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)
    bin_img[bin_img != 0] = 126
    bin_img[bin_img == 0] = fill
    bin_img[bin_img == 126] = 0
    return bin_img.astype(np.uint8)


# def segment_image_by_color(img):
#     img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     HSV_mask = cv2.inRange(img_HSV, (0, 0, 18), (179, 255, 255))
#     HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
#
#     img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#     YCrCb_mask = cv2.inRange(img_YCrCb, (18, 0, 0), (255, 255, 255))
#     YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
#
#     global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
#     global_mask = cv2.medianBlur(global_mask, 3)
#     global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
#     segmented_image = cv2.bitwise_and(img, img, mask=global_mask)
#     return segmented_image, global_mask


def extract_video_v2(model: SpringMetalDetector, class_id, video_path: Path, output_dir: Path = None):
    cap = cv2.VideoCapture(str(video_path))

    # base_filename = f"{video_path.stem}_frame"
    # print(f"Processing {video_path}")

    # # images_dir = output_dir / "images"
    # # labels_dir = output_dir / "labels"
    # # images_dir.mkdir(parents=True, exist_ok=True)
    # # labels_dir.mkdir(parents=True, exist_ok=True)
    # aspect_ratio = 1.3
    # aspect_ratio_tol = 0.2
    # min_area = 0.01
    # fgbg2 = cv2.createBackgroundSubtractorMOG2()
    # bg_image = cv2.imread("C:\\Users\\maftuh.mashuri\\Pictures\\Video_20240420185538514 - frame at 0m0s.jpg", 0)

    def _resize(im):
        return resize_image(im, width=640)
    

    def return_segment_mean_shift(image, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = image.shape[:2]
        result = np.zeros((h, w, 3), dtype=np.uint8) 
    
        result_bboxes = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

            if len(approx) <= 6:
                x, y, w, h = cv2.boundingRect(cnt)

                wn, hn = w / image.shape[1], h / image.shape[0]
                area = wn * hn

                if area < 0.01:
                    continue

                result_bboxes.append([x, y, x + w, y + h])

                result = cv2.drawContours(result, [cnt], -1, (255, 255, 255), 3)

        return result, result_bboxes

    def find_angle_of_white_box(frame):
        # Simpan nilai angle
        angle = 0
        
        # Konversi gambar ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Ambil kontur dari gambar
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter kontur untuk kotak putih
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 1000:  # Menyesuaikan dengan kebutuhan Anda
                # Dapatkan titik sudut
                pts = approx.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                
                # Atur urutan titik sesuai dengan urutan:
                # Top-Left, Top-Right, Bottom-Right, dan Bottom-Left
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                
                # Hitung lebar dan tinggi
                (tl, tr, br, bl) = rect
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                
                # Tentukan sisi yang lebih panjang
                if widthA > widthB:
                    long_side = br - bl
                else:
                    long_side = tr - tl
                
                # Hitung sudut rotasi
                angle = np.arctan2(long_side[1], long_side[0]) * 180 / np.pi
                
                # # Gambar garis yang menunjukkan sisi panjang
                # cv2.line(frame, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), (0, 0, 255), 2)
                
                # # Tampilkan sudut
                # cv2.putText(frame, f'Angle: {angle:.2f}', (int(tl[0]), int(tl[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                break
        
        return angle

        # return frame
    

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            cap.release()
            break

        # ori_frame = frame.copy()
        resize_frame = _resize(frame)

        """
        segmented, mask = segment_image_by_color(resize_image(frame, width=960))
        mask_ori = mask.copy()
        mask_ori = _resize(mask_ori)
        segmented = _resize(segmented)
        mask = _resize(mask)
        """

        results = model.detect(resize_frame)
        h, w = resize_frame.shape[:2]

        for result in results:
            if results is not None:
                cls, conf, xyxyn = result

                x1, y1, x2, y2 = expand_bbox(xyxyn, 0.15)
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)

                cropped_frame = resize_frame[y1:y2, x1:x2]

                _, cropped_mask = segment_image_by_color(cropped_frame)
                bbox_cropped_frame, bbox = return_segment_mean_shift(cropped_frame, cropped_mask)

                angle = find_angle_of_white_box(bbox_cropped_frame)
                # cv2.imshow("bbox cropped frame", bbox_cropped_frame)
                # if abs(angle) >= 20: angle = 0
                
                rot_frame = (rotate(cropped_frame, angle, resize=False) * 255).astype(np.uint8)

                cv2.imshow("cropped frame", cropped_frame)
                cv2.imshow("rot frame", rot_frame)

        segmented, mask = segment_image_by_color(resize_frame)
        # img, mask2, cnts = metal_detector(frame)
        # mask2 = _resize(mask2)
        mask_ori = mask.copy()
        segmented2, mask = segment_mean_shift(resize_frame, mask)

        # all_contours, hierarchy = cv2.findContours(
        #     mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        # )
        #
        # edges_contour =
        #
        # # Loop through individual contours
        # for contour in all_contours:
        #     # Approximate contour to a polygon
        #     perimeter = cv2.arcLength(contour, True)
        #     approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        #     x, y, w, h = cv2.boundingRect(approx)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(_resize(gray), (3, 3), sigmaX=0, sigmaY=0)
        # edges = cv2.Canny(blur, 100, 200)
        # blur = cv2.GaussianBlur(gray, (13, 13), 0)
        # ret, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
        # cv2.imshow('Segmentation', _resize(segmented))
        cv2.imshow('mask1', mask_ori)
        # cv2.imshow('mask2', mask2)
        # cv2.imshow('mask3', mask3)
        cv2.imshow('Segmentation 2', segmented2)
        # cv2.imshow('Edge', edges)
        #
        # cnts = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        # edges_contour = np.zeros_like(edges)
        # for c in cnts:
        #     cv2.drawContours(edges_contour, [c], 0, (0, 255, 0), 3)
        # cv2.imshow('Edge Contour', edges)

        # filename = f"{base_filename}_{str(time.time()).replace('.', '_')}.jpg"
        # bbox_filename = f"{Path(filename).stem}_bbox.jpg"
        # label_filename = f"{Path(filename).stem}.txt"

        # frame2, thresh, bboxes = detect_metal(frame, draw_bbox=True)

        # height, width = frame.shape[:2]
        #
        # for i, result in enumerate(results):
        #     cls, conf, xyxyn = result
        #     # xyxyn = expand_bbox(xyxyn, 0.05)
        #     # xyxyn = make_bbox_square(xyxyn)
        #     # frame = draw_bbox(frame, [cls, conf, xyxyn])
        #     x1, y1, x2, y2 = unormalize_bbox(xyxyn, frame.shape)
        #     cropped_frame = frame[y1:y2, x1:x2]
        #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #     h, s, v = cv2.split(hsv)
        # print("h", st.mode(h).index(0))
        # print("s", st.mode(s))
        # print("v", st.mode(v))
        # b, g, r = cv2.split(cropped_frame)
        # print("blue", np.unique(b))
        # print("green", np.unique(g))
        # print("r", np.unique(r))
        # print(np.unique(b), np.unique(g), np.unique(r))
        # # thresh = _create_mask(cropped_frame)
        # mask = np.zeros(cropped_frame.shape[:2], np.uint8)
        # bgdModel = np.zeros((1, 65), np.float64)
        # fgdModel = np.zeros((1, 65), np.float64)
        # rect = (50, 50, 450, 290)
        # cv2.grabCut(cropped_frame, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        # cropped_frame = cropped_frame * mask2[:, :, np.newaxis]
        #
        # cv2.imshow(f"cropped_frame_{i}", cropped_frame)

        # filtered_bboxes = []
        # for x, y, w, h in bboxes:
        # wn = w / width
        # hn = h / height
        # area_n = wn * hn
        # ratio = w / h
        # print(area_n, ratio)
        # if area_n < min_area or abs(ratio - aspect_ratio) > aspect_ratio_tol:
        #     continue
        # filtered_bboxes.append([x, y, x+w, y+h])

        # print(filtered_bboxes)
        # print(image, thresh)

        # with open(labels_dir / label_filename, "a") as f:
        #     x1, y1, x2, y2 = xyxyn
        #     x = (x1 + x2) / 2
        #     y = (y1 + y2) / 2
        #     w = abs(x2 - x1)
        #     h = abs(y2 - y1)
        #     f.write(f"{class_id} {x} {y} {w} {h}" + "\n")

        frame = resize_image(frame, width=640)
        # thresh = resize_image(thresh, width=640)
        # frame2 = resize_image(frame2, width=640)
        # ori_frame = resize_image(ori_frame, width=640)

        # if len(results) > 0:
        #     cv2.imwrite(str(images_dir / filename), ori_frame)
        #     cv2.imwrite(str(images_dir / bbox_filename), frame)

        cv2.imshow(video_path.name, frame)
        # cv2.imshow(f"{video_path.name}_boxed", frame2)
        # cv2.imshow(f"{video_path.name}_thresh", thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def stream_detect_video(model: SpringMetalDetector, video_path: Path):
    cap = cv2.VideoCapture(str(video_path))

    print(f"Processing {video_path}")
    track_history = defaultdict(lambda: [])

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            cap.release()
            break

        processed_frame = model.pre_process(frame)
        results = model.model.track(processed_frame, stream=False, device=model.device)
        annotated_frame = results[0].plot()
        print(results[0].boxes)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # for res in prediction:
        #     print(res)

        # frame = resize_image(frame, width=640)
        #
        # cv2.imshow(video_path.name, frame)
        #
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    path = "D:\\charlie\\repos\\sme-automation-inspection-internal\\tools\\yolov8.pt"
    model = SpringMetalDetector(
        path,
        max_num_detection=1,
        distance_thresholds=[0.3, 0.5],
        device="cpu"
    )
    video_path = "D:\\charlie\\repos\\sme-automation-inspection-internal\\sample\\good\\atas\\Video_20240420191017586.avi"
    extract_video_v2(model, 0, Path(video_path))
