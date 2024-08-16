from ultralytics import YOLO
import numpy as np
import cv2


# Define a class to change pixels outside object masks to white
class ChangeOutsideToWhite:
    def __init__(self, model_path='best.pt', img_size=640, confidence=0.5, device="cpu"):
        # Initialize YOLO model with a pretrained model
        self.model = YOLO(model_path)
        # Image size to be used in inference
        self.img_size = img_size
        # Confidence threshold for instance segmentation
        self.confidence = confidence
        self.model.to(device)
        self.device = device

    def remove_background(self, image):
        # Run inference with given arguments
        results = self.model.predict(
            image, save=False, imgsz=self.img_size, conf=self.confidence, show_labels=False)

        # Loop through segmentation results
        segmentation = None
        for r in results:
            # print(r)
            try:
                # Extract mask from segmentation results and convert to binary image format
                # masks = (r.masks.data[0].detach().numpy().astype(int) * 255).astype(np.uint8)
                masks = (r.masks.data[0].detach().cpu().numpy().astype(int) * 255).astype(np.uint8)
                # print(type(r.masks.xy[0]))
                segmentation = r.masks.xy[0].astype(int)
            except Exception as e:
                """
                # Creating a matrix of a specific size, filled with value 0
                height, width, _ = image.shape
                masks = np.zeros((height, width))
                """
                return None, None

        # Dilate the mask to extend it by 7 pixels
        kernel = np.ones((7, 7), np.uint8)
        dilated_mask = cv2.dilate(masks, kernel, iterations=1)

        """
        # Make the areas outside the dilated mask white
        image[dilated_mask == 0] = [255, 255, 255]

        return image
        """

        return dilated_mask, segmentation

    def return_image(self, masks_data, image):
        if masks_data is None:
            return None
            # Make image using height and width
            # height, width, _ = image.shape
            #
            # # Create a NumPy array with all elements set to white
            # white_image = np.ones((height, width, 3), dtype=np.uint8) * 255
            # return white_image

        else:
            # Make the areas outside the dilated mask white
            image[masks_data == 0] = [255, 255, 255]
            return image
