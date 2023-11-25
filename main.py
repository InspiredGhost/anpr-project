import numpy as np
from ultralytics import YOLO
import cv2

import util

from sort.sort import *

motion_tracker = Sort()

results = {}

# Load Models
coco_model = YOLO('yolov8n.pt')  # Yolo Model
license_plate_detector = YOLO(
    '/Users/inspiredghost/Documents/ModelData/code/runs/detect/train4/weights/best.pt')  # My Model

# Load Video
cap = cv2.VideoCapture('/Users/inspiredghost/Documents/ModelData/videos/20231111_131128.mp4')

items = [0, 1]

# Read Frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret and frame_nmr < 2000:
        # if ret :
        results[frame_nmr] = {}
        print("Frame #", frame_nmr)
        # detect vehicles
        detections = license_plate_detector(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in items:
                detections_.append([x1, y1, x2, y2, score])

        # Track Cars
        print(detections_)
        if detections_ is not None and len(detections_) > 0:
            track_ids = motion_tracker.update(np.asarray(detections_))
        # else:
        # track_ids = motion_tracker.update(np.empty((0, 5)))

        # Detect Licence Plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            #  Assign Plate To Car
            xcar1, ycar1, xcar2, ycar2, car_id = util.get_car(license_plate, track_ids)

            # Crop License Plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            # Process Licence Plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_threshold = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # cv2.imshow("Original_Crop", license_plate_crop)
            # cv2.imshow("Thresh_Crop", license_plate_crop_threshold)

            # Read Licence Plate
            licence_plate_text, license_plate_score = util.read_license_plate(license_plate_crop_threshold)

            if licence_plate_text is not None:
                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                              'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                'text': licence_plate_text,
                                                                'bbox_score': score,
                                                                'text_score': license_plate_score}}

# Write Results
util.write_csv(results, './test.csv')
