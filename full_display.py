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

# Nouvelles connexions basées sur le tableau filtré de 16 points
CUSTOM_BODY_CONNECTIONS = [
    (0, 1),   # Épaules
    (0, 2), (2, 4), (4, 6), # Bras gauche
    (1, 3), (3, 5), (5, 7), # Bras droit
    (0, 8), (1, 9), # Torse (Épaules -> Hanches)
    (8, 9),   # Hanches
    (8, 10), (10, 12), (12, 14), # Jambe gauche
    (9, 11), (11, 13), (13, 15)  # Jambe droite
]

# Variables
color_landmarks_hands = (255, 0, 0)
thickness_landmarks_hands = 3
circle_radius_landmarks_hands = 3

color_landmarks_body = (0, 0, 255)
thickness_landmarks_body = 3
circle_radius_landmarks_body = 3

color_connection_hands = (0, 255, 0)
thickness_connection_hands = 3

color_connection_body = (255, 255, 255)
thickness_connection_body = 3

color_custom_connection_hands = (0, 255, 255)
color_custom_connection_two_hands = (0, 255, 255)
color_custom_connection_body = (0, 255, 255)

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


def extract_pose_landmarks(results):
    coords = []

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])

    return np.array(coords, dtype=np.float32)

def delete_coords(pose_array):
    keep = [False] * 33

    # Liste des repères à garder
    keep[11] = keep[12] = True  # Épaules
    keep[13] = keep[14] = True  # Coudes
    keep[15] = keep[16] = True  # Poignets
    keep[19] = keep[20] = True  # Index (Mains)
    keep[23] = keep[24] = True  # Hanches
    keep[25] = keep[26] = True  # Genoux
    keep[27] = keep[28] = True  # Chevilles
    keep[31] = keep[32] = True  # Pieds

    filtered = pose_array[keep]
    return filtered

def normalize_coords(filtered_array):
    # Calculate the center coordonate
    hip_center = (filtered_array[8] + filtered_array[9]) / 2
    # Calculate the shoulder width for scaling
    shoulder_center = (filtered_array[0] + filtered_array[1]) / 2
    # Calculate  the torso length
    torso_length = np.linalg.norm(shoulder_center - hip_center)

    # If torso is lost
    if torso_length < 1e-6:
        torso_length = 1.0

    # Normalize each point
    normalized_array = (filtered_array - hip_center) / torso_length

    return normalized_array

def clean_data(pose_array):
    filtered_array = delete_coords(pose_array)
    normalized_array = normalize_coords(filtered_array)
    return normalized_array

def draw_normalized_view(pose_array, normalized_view):
    # Récupère les 16 points prêts pour le ML
    normalized_pose_array = clean_data(pose_array)

    h, w, _ = normalized_view.shape
    scale = min(w, h) * 0.25
    offset_x = w // 2
    offset_y = h // 2

    # Dessiner les points
    for p in normalized_pose_array:
        x = int(p[0] * scale + offset_x)
        y = int(p[1] * scale + offset_y)
        cv.circle(normalized_view, (x, y), 5, (0, 255, 0), -1)

    # Dessiner les connexions avec la carte personnalisée
    for a, b in CUSTOM_BODY_CONNECTIONS:
        x1 = int(normalized_pose_array[a][0] * scale + offset_x)
        y1 = int(normalized_pose_array[a][1] * scale + offset_y)

        x2 = int(normalized_pose_array[b][0] * scale + offset_x)
        y2 = int(normalized_pose_array[b][1] * scale + offset_y)

        cv.line(normalized_view, (x1, y1), (x2, y2), (255, 255, 255), 2)

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

        # Create 3 separate render buffers
        # Resize reference (important so everything matches)
        h, w, _ = frame.shape

        # 1) Camera view (top-left)
        camera_view = frame.copy()

        # 2) Skeleton-only view (bottom-left)
        skeleton_view = np.zeros((h, w, 3), dtype=np.uint8)

        # 3) Normalized view (top-right)
        normalized_view = np.zeros((h, w, 3), dtype=np.uint8)

        # 4) 3D placeholder view (bottom-right)
        model_view = np.zeros((h, w, 3), dtype=np.uint8)

        # If body is detected, draw landmarks and connections on the frame
        if body_detected.pose_landmarks:
            # Extract the coordonates
            pose_array = extract_pose_landmarks(body_detected)

            # Print the coordonates
            # print(pose_array)
            # Print specific coordonates
            # print(pose_array[1])

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

            draw_normalized_view(pose_array, normalized_view)

            drawing.draw_landmarks(
                model_view,
                body_detected.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=custom_landmark_style_body,
                connection_drawing_spec=custom_connection_style_body
            )

            draw_custom_links_body(body_detected.pose_landmarks, camera_view)
            draw_custom_links_body(body_detected.pose_landmarks, skeleton_view)
            draw_custom_links_body(body_detected.pose_landmarks, model_view)

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

                drawing.draw_landmarks(
                    model_view,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=custom_landmark_style_hands,
                    connection_drawing_spec=custom_connection_style_hands
                )

                draw_custom_links_hands(hand_landmarks, camera_view)
                draw_custom_links_hands(hand_landmarks, skeleton_view)
                draw_custom_links_hands(hand_landmarks, model_view)

        # Draw link between the two hands
        hands_list = hands_detected.multi_hand_landmarks

        if hands_list is not None and len(hands_list) == 2:
            draw_hand_to_hand_connection(
                hands_list[0],
                hands_list[1],
                camera_view
            )
            draw_hand_to_hand_connection(
                hands_list[0],
                hands_list[1],
                skeleton_view
            )
            draw_hand_to_hand_connection(
                hands_list[0],
                hands_list[1],
                model_view
            )

        # Resize all to same height
        # Left = Horizontal
        # Right = Vertical
        display_w = int(1920 / 2)
        display_h = int(1080 / 2)
        camera_view = cv.resize(camera_view, (display_w, display_h))
        skeleton_view = cv.resize(skeleton_view, (display_w, display_h))
        normalized_view = cv.resize(normalized_view, (display_w, display_h))
        model_view = cv.resize(model_view, (display_w, display_h))

        # Stack left side (camera + skeleton)
        left_side = np.vstack((camera_view, skeleton_view))
        right_side = np.vstack((normalized_view, model_view))

        # Combine with right side (model)
        final = np.hstack((left_side, right_side))

        cv.imshow(window_name, final)

        # Exit the loop if 'q' key is pressed
        if cv.waitKey(20) & 0xFF == ord("q"):
            break

    cam.release()
    cv.destroyAllWindows()

bodyandhands()