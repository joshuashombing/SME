import cv2

cap = cv2.VideoCapture("C:\\Users\\sakti.prawira\\Documents\\output_demo.avi")

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)

path = "SlowedVideo.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(path, fourcc, 5, (width, height))

while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    output.write(frame)
    k = cv2.waitKey(24)
    if k == ord("q"):
        break

cap.release()
output.release()
cv2.destroyAllWindows()
