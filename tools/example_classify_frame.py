import cv2
from spring_classify import SpringClassifier as classify
import datetime

# Make classifier instance
classify = classify()

# Iterate 10 times
for i in range(1, 11):
    # Open a video capture object (0 for webcam, or filename for video file)
    cap = cv2.VideoCapture(1)  # Change to 0 for webcam
    
    # Get prediction
    prediction = classify.classify_frame(cap)

    # Print prediction with time
    print(i, prediction, datetime.datetime.now().strftime("%H:%M:%S.%f")[:-4])

# Release the video capture object
cap.release()
