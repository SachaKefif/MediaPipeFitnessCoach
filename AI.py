import cv2 as cv
import mediapipe as mp
import numpy as np

import Data
import Visualization

# Force NumPy to print standard numbers instead of scientific notation
np.set_printoptions(suppress=True)


def extract_data():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cam = cv.VideoCapture(1)

    if not cam.isOpened():
        print("Camera not found")
        return

    print("Starting data extraction...")

    while True:
        success, frame = cam.read()
        if not success:
            print("Camera frame not available")
            break

        h, w, _ = frame.shape
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        body_detected = pose.process(rgb_frame)

        if body_detected.pose_landmarks:
            # 1. You get the pose_array here
            pose_array = Visualization.extract_pose_landmarks(body_detected, w, h)

            # 2. Immediately pass it to clean_data()
            final_features = Data.clean_data(pose_array)

            # 3. Print it
            print("Extracted feature array :")
            print(final_features)

    cam.release()
    pose.close()

extract_data()
