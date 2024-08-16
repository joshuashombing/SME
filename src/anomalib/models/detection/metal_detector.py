import cv2


def detect_metal_raw(image):
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
    bboxes = []
    for c in cnts:
        box = cv2.boundingRect(c)
        bboxes.append(box)

    return bboxes, thresh


def detect_metal(image, aspect_ratio=1.3, aspect_ratio_tol=0.2, min_area=0.01, draw_bbox=False, verbose=True):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (13, 13), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7)

    # Morph close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours, sort for largest contour, draw contour
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    bboxes = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        wn = w / width
        hn = h / height
        area_n = wn * hn
        ratio = w / h
        if area_n < min_area or abs(ratio - aspect_ratio) > aspect_ratio_tol:
            if verbose:
                print(f"[FAILED] Invalid values: ratio={ratio}, normalize area={area_n}")
            continue

        if verbose:
            print(f"[SUCCESS] Valid values: ratio={ratio}, normalize area={area_n}")
        bboxes.append([x, y, x + w, y + h])

        if draw_bbox:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image, thresh, bboxes
