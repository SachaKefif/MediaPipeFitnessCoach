import cv2 as cv
import mediapipe as mp
import numpy as np

# Do we want to show only the skeleton ?
show_camera = True

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
        (8, 12),
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

    cam = cv.VideoCapture(0)

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

        # Invert the frame to mirror
        frame = cv.flip(frame, 1)

        # Convert the frame from BGR to RGB (required by MediaPipe)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        body_detected = pose.process(rgb_frame)
        hands_detected = hands.process(rgb_frame)

        # Create the black image if we don't want to see the camera
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        # If body is detected, draw landmarks and connections on the frame
        if body_detected.pose_landmarks:
            if show_camera:
                drawing.draw_landmarks(
                    frame,
                    body_detected.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=custom_landmark_style_body,
                    connection_drawing_spec=custom_connection_style_body
                )
            else:
                drawing.draw_landmarks(
                    canvas,
                    body_detected.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=custom_landmark_style_body,
                    connection_drawing_spec=custom_connection_style_body
                )

            if show_camera:
                draw_custom_links_body(body_detected.pose_landmarks, frame)
            else:
                draw_custom_links_body(body_detected.pose_landmarks, canvas)

        # If hands are detected, draw landmarks and connections on the frame
        if hands_detected.multi_hand_landmarks:
            for hand_landmarks in hands_detected.multi_hand_landmarks:
                if show_camera:
                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=custom_landmark_style_hands,
                        connection_drawing_spec=custom_connection_style_hands
                    )
                else:
                    drawing.draw_landmarks(
                        canvas,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=custom_landmark_style_hands,
                        connection_drawing_spec=custom_connection_style_hands
                    )

                if show_camera:
                    draw_custom_links_hands(hand_landmarks, frame)
                else:
                    draw_custom_links_hands(hand_landmarks, canvas)

        # Draw link between the two hands
        hands_list = hands_detected.multi_hand_landmarks
        if hands_list is not None and len(hands_list) == 2:
            hand1 = hands_list[0]
            hand2 = hands_list[1]

            if show_camera:
                draw_hand_to_hand_connection(hand1, hand2, frame)
            else:
                draw_hand_to_hand_connection(hand1, hand2, canvas)

        # Display every frame, even when no hand is currently detected
        if show_camera:
            cv.imshow(window_name, frame)
        else:
            cv.imshow(window_name, canvas)

        # Exit the loop if 'q' key is pressed
        if cv.waitKey(20) & 0xFF == ord("q"):
            break

    cam.release()
    cv.destroyAllWindows()

bodyandhands()