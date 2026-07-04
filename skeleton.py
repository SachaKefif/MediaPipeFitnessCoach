# This file will reproduce the nodes but without the camera displayed

import cv2 as cv
import mediapipe as mp
import numpy as np

# Body
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Hands
mp_hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles


def skeleton():
    # Body
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    # Hands
    hands = mp_hands.Hands(
        static_image_mode=False,  # Set to False for processing video frames
        max_num_hands=2,  # Maximum number of hands to detect
        min_detection_confidence=0.5,  # Minimum confidence threshold for hand detection
        min_tracking_confidence=0.5,
    )

    cam = cv.VideoCapture(0)

    if not cam.isOpened():
        print("Camera not found")
        return

    window_name = "Skeleton Only"
    cv.namedWindow(window_name)

    # Constantly run
    while True:
        # Read a frame from the camera
        success, frame = cam.read()
        # If the frame is not available, stop the program
        if not success:
            print("Camera frame not available")
            break

        # Convert the frame from BGR to RGB (required by MediaPipe)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        body_detected = pose.process(rgb_frame)
        hands_detected = hands.process(rgb_frame)

        # Create empty image with no camera
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        # If body is detected, draw landmarks and connections on the frame
        if body_detected.pose_landmarks:
            mp_drawing.draw_landmarks(
                canvas,
                body_detected.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_styles.get_default_pose_landmarks_style()
            )

        # If hands are detected, draw landmarks and connections on the frame
        if hands_detected.multi_hand_landmarks:
            for hand_landmarks in hands_detected.multi_hand_landmarks:
                drawing.draw_landmarks(
                    canvas,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )

        # Display every frame, even when no hand is currently detected
        cv.imshow(window_name, canvas)

        # Exit the loop if 'q' key is pressed
        if cv.waitKey(20) & 0xFF == ord("q"):
            break

    cam.release()
    cv.destroyAllWindows()

skeleton()