# This file will display 3 things :
# Upper left corner : video + landmarks
# Lower left corner : landmarks
# Right : 3d model + landmarks

import cv2 as cv
import mediapipe as mp
import numpy as np

# Body
mp_pose = mp.solutions.pose
# Hands
mp_hands = mp.solutions.hands

# General
drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

# Variables
color_landmarks_hands = (0, 0, 255)
thickness_landmarks_hands = 3
circle_radius_landmarks_hands = 3

color_landmarks_body = (255, 0, 0)
thickness_landmarks_body = 3
circle_radius_landmarks_body = 3

color_connection_hands = (255, 255, 255)
thickness_connection_hands = 3

color_connection_body = (0, 255, 0)
thickness_connection_body = 3

color_custom_connection_hands = (0, 255, 255)
color_custom_connection_two_hands = (0, 255, 255)
color_custom_connection_body = (255, 0, 255)

# Custom Landmark Styles
custom_landmark_style_hands = drawing.DrawingSpec(
    color=color_landmarks_hands,
    thickness=thickness_landmarks_hands,
    circle_radius=circle_radius_landmarks_hands
)
custom_landmark_style_body = drawing.DrawingSpec(
    color=color_landmarks_body,
    thickness=thickness_landmarks_body,
    circle_radius=circle_radius_landmarks_body
)

# Custom Connection Styles
custom_connection_style_hands = drawing.DrawingSpec(
    color=color_connection_hands,
    thickness=thickness_connection_hands
)
custom_connection_style_body = drawing.DrawingSpec(
    color=color_connection_body,
    thickness=thickness_connection_body
)

# Custom links

# Hands
def draw_custom_links_hands(hand_landmarks, image, color=color_custom_connection_hands, thickness=thickness_connection_hands):
    h, w, _ = image.shape

    def point(id):
        lm = hand_landmarks.landmark[id]
        return int(lm.x * w), int(lm.y * h)

    # Standard hand structure + extra freedom to customize
    connections = [
        # Custom Links
        (4, 8)
    ]

    for a, b in connections:
        cv.line(image, point(a), point(b), color, thickness)

    # Optional: highlight joints (nodes)
    for i in range(21):
        cv.circle(image, point(i), 3, (0, 0, 255), -1)

# Body
def draw_custom_links_body(landmarks, image, color=color_custom_connection_body, thickness=thickness_connection_body):
    h, w, _ = image.shape

    def point(id):
        lm = landmarks.landmark[id]
        return int(lm.x * w), int(lm.y * h)

    # Example custom links
    connections = [
        # Custom Links
        (10, 12), (9, 11)
    ]

    for a, b in connections:
        cv.line(image, point(a), point(b), color, thickness)

# Two hands
def draw_hand_to_hand_connection(hand1, hand2, image, color=color_custom_connection_two_hands, thickness=thickness_connection_hands):
    h, w, _ = image.shape

    def point(hand, idx):
        lm = hand.landmark[idx]
        return int(lm.x * w), int(lm.y * h)

    connections = [
        # Custom Links
        (12, 12),
    ]

    for a, b in connections:
        cv.line(image, point(hand1, a), point(hand2, b), color, thickness)

def bodyandhands():
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

    cam = cv.VideoCapture(1)

    if not cam.isOpened():
        print("Camera not found")
        return

    window_name = "Body and Hands Detection"
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

        # Create the black image if we don't want to see the camera
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Create 3 separate render buffers
        # Resize reference (important so everything matches)
        h, w, _ = frame.shape

        # 1) Camera view (top-left)
        camera_view = frame.copy()

        # 2) Skeleton-only view (bottom-left)
        skeleton_view = np.zeros((h, w, 3), dtype=np.uint8)

        # 3) 3D placeholder view (right side)
        model_view = np.zeros((h, w, 3), dtype=np.uint8)

        # If body is detected, draw landmarks and connections on the frame
        if body_detected.pose_landmarks:
            drawing.draw_landmarks(
                camera_view,
                body_detected.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=custom_landmark_style_body,
                connection_drawing_spec=custom_connection_style_body
            )

            drawing.draw_landmarks(
                skeleton_view,
                body_detected.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=custom_landmark_style_body,
                connection_drawing_spec=custom_connection_style_body
            )

            draw_custom_links_body(body_detected.pose_landmarks, camera_view)
            draw_custom_links_body(body_detected.pose_landmarks, skeleton_view)

        # If hands are detected, draw landmarks and connections on the frame
        if hands_detected.multi_hand_landmarks:
            for hand_landmarks in hands_detected.multi_hand_landmarks:
                # always draw camera view if it exists
                drawing.draw_landmarks(
                    camera_view,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=custom_landmark_style_hands,
                    connection_drawing_spec=custom_connection_style_hands
                )

                # always draw skeleton view
                drawing.draw_landmarks(
                    skeleton_view,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=custom_landmark_style_hands,
                    connection_drawing_spec=custom_connection_style_hands
                )

                draw_custom_links_hands(hand_landmarks, camera_view)
                draw_custom_links_hands(hand_landmarks, skeleton_view)

        # Draw link between the two hands
        hands_list = hands_detected.multi_hand_landmarks

        if hands_list is not None and len(hands_list) == 2:
            draw_hand_to_hand_connection(
                hands_list[0],
                hands_list[1],
                skeleton_view
            )
            draw_hand_to_hand_connection(
                hands_list[0],
                hands_list[1],
                camera_view
            )

        # Resize all to same height
        camera_view = cv.resize(camera_view, (640, 360))
        skeleton_view = cv.resize(skeleton_view, (640, 360))
        model_view = cv.resize(model_view, (640, 720))

        # Stack left side (camera + skeleton)
        left_side = np.vstack((camera_view, skeleton_view))

        # Combine with right side (model)
        final = np.hstack((left_side, model_view))

        cv.imshow(window_name, final)

        # Exit the loop if 'q' key is pressed
        if cv.waitKey(20) & 0xFF == ord("q"):
            break

    cam.release()
    cv.destroyAllWindows()

bodyandhands()