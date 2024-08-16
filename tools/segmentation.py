import cv2
import numpy as np


def do_nothing(x):
    pass


def detect_circle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 5)  # cv2.bilateralFilter(gray,10,50,50)

    minDist = 100
    param1 = 30  # 500
    param2 = 50  # 200 #smaller value-> more false circles
    minRadius = 50
    maxRadius = 100  # 10

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1, minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

    return img


# create slider here
cv2.namedWindow("Slider")
cv2.resizeWindow("Slider", 640, 480)
cv2.createTrackbar("Hue Min", "Slider", 80, 255, do_nothing)
cv2.createTrackbar("Hue Max", "Slider", 125, 255, do_nothing)
cv2.createTrackbar("Saturation Min", "Slider", 0, 255, do_nothing)
cv2.createTrackbar("Saturation Max", "Slider", 255, 255, do_nothing)
cv2.createTrackbar("Value Min", "Slider", 0, 255, do_nothing)
cv2.createTrackbar("Value Max", "Slider", 255, 255, do_nothing)

img = cv2.imread("./ok (10).jpg")
img = cv2.resize(img, (640, 480))

while True:

    hue_min = cv2.getTrackbarPos("Hue Min", "Slider")
    hue_max = cv2.getTrackbarPos("Hue Max", "Slider")
    sat_min = cv2.getTrackbarPos("Saturation Min", "Slider")
    sat_max = cv2.getTrackbarPos("Saturation Max", "Slider")
    val_min = cv2.getTrackbarPos("Value Min", "Slider")
    val_max = cv2.getTrackbarPos("Value Max", "Slider")

    # set bounds
    lower_bound = np.array([hue_min, sat_min, val_min])
    upper_bound = np.array([hue_max, sat_max, val_max])

    # convert to HSV image
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # create mask
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # invert morp image
    mask = morph

    img_show = img.copy()
    cnts = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    big_contour = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(big_contour, True)
    big_contour = cv2.approxPolyDP(big_contour, 0.001 * peri, True)
    peri = cv2.arcLength(big_contour, True)
    big_contour = cv2.approxPolyDP(big_contour, 0.001 * peri, True)
    cv2.drawContours(img_show, [big_contour], -1, (36, 255, 12), 2)

    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # img_show = img.copy()
    # for c in cnts:
    #     cv2.drawContours(img_show, [c], -1, (36, 255, 12), 2)
    #     break

    # apply mask to image
    # result = cv2.bitwise_and(img_show, img, mask=mask)

    resulting_img = cv2.bitwise_and(img, img, mask=mask)
    img_circled = detect_circle(resulting_img.copy())

    stacked_imgs = np.hstack([img_show, img_circled])

    #     create a stacked image of the original and the HSV one.
    cv2.imshow("Image", stacked_imgs)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
