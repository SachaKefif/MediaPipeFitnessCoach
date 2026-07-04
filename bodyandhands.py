import cv2 as cv
import mediapipe as mp
import numpy as np

# Do we want to show only the skeleton ?
show_camera = False

# Body
mp_pose = mp.solutions.pose
# Hands
mp_hands = mp.solutions.hands

# General
drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

# Custom Landmark Styles
custom_landmark_style_hands = drawing.DrawingSpec(
    color=(0, 0, 255),
    thickness=3,
    circle_radius=3
)
custom_landmark_style_body = drawing.DrawingSpec(
    color=(255, 0, 0),
    thickness=3,
    circle_radius=3
)

# Custom Connection Styles
custom_connection_style_hands = drawing.DrawingSpec(
    color=(255, 255, 255),
    thickness=3
)
custom_connection_style_body = drawing.DrawingSpec(
    color=(0, 255, 0),
    thickness=3
)

# Custom links

# Hands
def draw_custom_links_hands(hand_landmarks, canvas, color=(0, 255, 255), thickness=2):
    h, w, _ = canvas.shape

    def point(id):
        lm = hand_landmarks.landmark[id]
        return int(lm.x * w), int(lm.y * h)

    # Standard hand structure + extra freedom to customize
    connections = [
        # Custom Links
        (4, 8)
    ]

    for a, b in connections:
        cv.line(canvas, point(a), point(b), color, thickness)

    # Optional: highlight joints (nodes)
    for i in range(21):
        cv.circle(canvas, point(i), 3, (0, 0, 255), -1)

# Body
def draw_custom_links_body(landmarks, image):
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
        cv.line(image, point(a), point(b), (255, 0, 255), 2)

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

        # Convert the frame from BGR to RGB (required by MediaPipe)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        body_detected = pose.process(rgb_frame)
        hands_detected = hands.process(rgb_frame)

        # Create the black image only if we don't want to see the camera
        if not show_camera:
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

                draw_custom_links_hands(hand_landmarks, canvas)

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