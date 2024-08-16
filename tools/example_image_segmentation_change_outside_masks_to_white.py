from image_segmentation_change_outside_masks_to_white import ChangeOutsideToWhite as background_changer
import cv2
import numpy as np

# Make background changer instance
background_changer = background_changer(
    model_path="D:\\Projects\\SME\\anomalib\\tools\\best.pt"
)

# Open a connection to the USB camera
cap = cv2.VideoCapture(0)  # Change to 0 for webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Clone the original frame
        original_frame = frame.copy()

        # Apply the changes using the method in the ChangeOutsideToWhite class
        masks = background_changer.remove_background(frame)

        # Return image from remove_background() function
        result_image = background_changer.return_image(masks, frame)

        # Stack the original frame and the processed frame horizontally
        stacked_frame = np.hstack((original_frame, result_image))

        # Display the stacked frame
        cv2.imshow('Original vs Processed', stacked_frame)

        # Print masks data
        print("masks data:", masks)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()