from ultralytics import YOLO
import cv2
import numpy as np
import math
import os
from pathlib import Path
import time

def resize_shape(source_shape, width=None, height=None):
    (h, w) = source_shape

    if width is None and height is None:
        return h, w

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        shape = (height, int(w * r))

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        shape = (int(h * r), width)

    return shape


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA) -> np.ndarray:
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    h, w = resize_shape(source_shape=image.shape[:2], width=width, height=height)

    # resize the image
    resized = cv2.resize(image, (w, h), interpolation=inter)

    # return the resized image
    return resized


def get_rect_points(results):
    rectangles = []  # List to store rectangle corner points

    if results:
        for xy in results[0].masks.xy:
            contour = xy.reshape((-1, 1, 2)).astype(np.int32)
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            
            # Go to next loop is approx is None
            if approx is None: continue

            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Check box values
            # Modify the box to replace negative numbers with 0
            new_box = [[max(0, num) for num in sub_box] for sub_box in box]

            # Append rectangle corner points to the list
            rectangles.append(new_box)
            
    return rectangles


def get_angle_from_rect_points(points):
    angles = []  # List to store rectangle angles
    for point in points:
        # Sort points
        point = sorted(point, key=lambda p:p[1])
        point = sorted(point[:2], key=lambda p:p[0])

        # Extract coordinates of the first point and second point
        point1, point2 = point[:2]
        x1, y1 = point1
        x2, y2 = point2

        # Calculate the differences in y (dy) and x (dx) coordinates between the two points
        dy = y2 - y1
        dx = x2 - x1

        # Calculate the arctangent of dy/dx (angle in radians)
        angle_radians = math.atan2(dy, dx)

        # Convert angle from radians to degrees
        angle_degrees = math.degrees(angle_radians)

        # # Ensure the angle is within the range of 0-360 degrees
        # if angle_degrees < 0:
        #     angle_degrees += 360

        # Append rectangle corner points to the list
        angles.append(angle_degrees)

    return angles


def resize_bbox(resize, x, y, width, height):
    # Calculate the center coordinates of the original bounding box
    center_x = x + width / 2
    center_y = y + height / 2

    # Calculate the new width and height (increase by 10%)
    new_width = int(width * resize)
    new_height = int(height * resize)

    # Calculate the new top-left corner coordinates based on the resized dimensions
    new_x = int(center_x - new_width / 2)
    new_y = int(center_y - new_height / 2)

    # Return the resized bounding box coordinates
    return new_x, new_y, new_width, new_height


def crop_frame_with_center(frame, new_width, new_height):
    # Get the dimensions of the rotated_image
    rotated_image_height, rotated_image_width = frame.shape[:2]

    # Get center image
    center_x, center_y = rotated_image_width / 2, rotated_image_height / 2

    # Calculate top-left corner coordinates for cropping
    x_min = int(center_x - new_width / 2)
    y_min = int(center_y - new_height / 2)

    # Ensure coordinates are within the frame boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame.shape[1], x_min + new_width)
    y_max = min(frame.shape[0], y_min + new_height)

    # Crop the frame using the calculated coordinates
    cropped_frame = frame[y_min:y_max, x_min:x_max]

    return cropped_frame


def crop_frame_with_angle(frame, points):
    frames = []  # List to store rotated frames

    # Get angles from rectangle points
    angles = get_angle_from_rect_points(points)

    for i, point in enumerate(points):
        # Convert the list of rectangle corner points to NumPy array
        rect_points = np.array(point, dtype=np.float32)

        # Determine the bounding box of the rectangle
        x, y, w, h = cv2.boundingRect(rect_points)

        # Resize bbox 1.2 larger
        x, y, w, h = resize_bbox(1.2, x, y, w, h)

        # Crop the region of interest (ROI) from the image using NumPy array slicing
        cropped_frame = frame[y:y+h, x:x+w]

        # Get the dimensions of the image
        height, width = cropped_frame.shape[:2]

        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angles[i], 1)

        # Apply the rotation matrix to the image using warpAffine
        rotated_image = cv2.warpAffine(cropped_frame, rotation_matrix, (width, height))

        # Crop frame with center
        new_height, new_width = 560, 680
        rotated_cropped_image = crop_frame_with_center(rotated_image, new_width, new_height)

        # Append rotated frames to the list
        frames.append(rotated_cropped_image)

    return frames


def draw_line(image, points, color=(0, 0, 255), thickness=2):
    # Extract coordinates of the first point and second point
    point1, point2 = points[:2]
    """
    Menggambar garis antara dua titik pada gambar.

    Args:
        image (numpy.ndarray): Gambar input.
        point1 (tuple): Koordinat titik pertama (x1, y1).
        point2 (tuple): Koordinat titik kedua (x2, y2).
        color (tuple): Warna garis dalam format BGR (default: merah).
        thickness (int): Ketebalan garis (default: 2).

    Returns:
        numpy.ndarray: Gambar dengan garis yang digambar.
    """
    # Gambar garis pada gambar input
    output_image = image.copy()
    cv2.line(output_image, point1, point2, color, thickness)

    return output_image


def read_files(directory, extension):
    files = []  # Membuat daftar kosong untuk nama file

    # Membaca semua file dalam direktori
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Memeriksa apakah file adalah file
        if os.path.isfile(file_path) and filename.endswith(extension):
            files.append(filename)  # Menambahkan nama file ke dalam daftar

    return files


def is_bbox_center(bbox, threshold):
    center = 0.5

    if isinstance(threshold, int):
        threshold = [threshold, threshold]

    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    is_center = abs(center_x - center) <= threshold[0] and abs(center_y - center) <= threshold[1]
    distance = np.sqrt((center_x - center) ** 2 + (center_y - center) ** 2)
    return is_center, distance


def select_closest_bbox(results):
    conf_threshold = 0.5
    max_num_detection = 5
    thresholds = [0.35, 0.5]

    filtered_results = []
    for result in results:
        conf = result.boxes.conf.detach().item()
        if conf < conf_threshold:
            continue

        xyxyn = result.boxes.xyxyn.detach().squeeze().cpu().tolist()
        is_center, distance = is_bbox_center(xyxyn, thresholds)

        if is_center:
            filtered_results.append((distance, result))

    if len(filtered_results) == 0:
        return []

    sorted_results = sorted(filtered_results, key=lambda x: x[0])
    return [res[1] for res in sorted_results[:max_num_detection]]



if __name__ == "__main__":
    # Load a model
    model = YOLO('..\\runs30-04-2024\\runs\\segment\\train\\weights\\best.pt')  # load a custom model

    # Set all path
    vid_data_path = Path("D:\\charlie\\repos\\sme-segmentation\\datasets-SME-30-04-2024\\datasets")
    frame_save_path = Path("D:\\charlie\\repos\\sme-segmentation\\datasets-SME-30-04-2024\\frame")

    # Read all files in json_path
    vid_data_lists = read_files(vid_data_path, 'avi')
    print(vid_data_lists)


    for vid_data_list in vid_data_lists:
        print("\nVideo name:", vid_data_list)

        # Open the video file
        cap = cv2.VideoCapture(str(Path(vid_data_path / vid_data_list)))

        # Check if the video file is successfully opened
        if not cap.isOpened():
            print("Error: Could not open video file.")
            exit()

        # Get name from video name
        vid_name_without_extension = os.path.splitext(vid_data_list)[0]

        # Make folder
        frame_save_path_local = Path(frame_save_path/ vid_name_without_extension)
        os.makedirs(frame_save_path_local, exist_ok=True)

        # Read until video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Break loop, if not ret
            if not ret: break
 
            # Predict with the model
            results = model.predict(frame, verbose=True, stream=False, device="cpu")  # predict on an image

            # Check there are any objects in the frame
            if results[0].masks is not None:
                # Filter result
                results = select_closest_bbox(results[0])

                # Get rectangle points
                rect_points = get_rect_points(results)

                # Get rotated frames
                rotated_frames = crop_frame_with_angle(frame, rect_points)

                # Get the current time as a timestamp
                timestamp = time.time()

                # Show or save rotated frames
                for i, rotated_frame in enumerate(rotated_frames):
                    # Set frame name
                    frame_name = str(Path(frame_save_path_local / f"{vid_name_without_extension}_{timestamp}_frame {i+1}.jpg"))

                    # cv2.imshow(f"Frame {i+1}", rotated_frame)
                    cv2.imwrite(frame_name, rotated_frame)
                
                # # Show objects with rect in frame
                # for rect_point in rect_points:
                #     frame = draw_line(frame, rect_point)
                # # cv2.drawContours(frame, rect_points, 0, (0, 255, 0), thickness=2)  # Green contour
                # frame = resize_image(frame, width=440)
                # cv2.imshow('Lines frame', frame)

                # # Print angles from rectangle points, will not effect other var
                # print(get_angle_from_rect_points(rect_points))

                # # Get results frame
                # result_frame = resize_image(results[0].plot(), width=440)
                # cv2.imshow('Result frame', result_frame)

                # # Wait for imshow
                # cv2.waitKey(0)
            
            # if there no objects in the frame
            else:
                print("results[0].masks is None\n")

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()
