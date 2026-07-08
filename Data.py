import cv2 as cv
import mediapipe as mp
import numpy as np

# Force NumPy to print standard numbers instead of scientific notation
np.set_printoptions(suppress=True)

# Normalized body coordonates
L_SHOULDER, R_SHOULDER = 0, 1
L_ELBOW, R_ELBOW = 2, 3
L_WRIST, R_WRIST = 4, 5
L_HAND, R_HAND = 6, 7
L_HIP, R_HIP = 8, 9
L_KNEE, R_KNEE = 10, 11
L_ANKLE, R_ANKLE = 12, 13
L_FOOT, R_FOOT = 14, 15

# Nouvelles connexions basées sur le tableau filtré de 16 points
CUSTOM_BODY_CONNECTIONS = [
    (L_SHOULDER, R_SHOULDER),   # Épaules
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (L_WRIST, L_HAND), # Bras gauche
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (R_WRIST, R_HAND), # Bras droit
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), # Torse (Épaules -> Hanches)
    (L_HIP, R_HIP),   # Hanches
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (L_ANKLE, L_FOOT), # Jambe gauche
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE), (R_ANKLE, R_FOOT)  # Jambe droite
]

def extract_pose_landmarks(results, w, h):
    coords = []

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            # Multiply by width and height to restore a perfect mathematical grid
            coords.append([lm.x * w, lm.y * h, lm.z * w])

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

# Calculate the angle between three landmarks
def calc_angle(landmarks, p1_idx, p2_idx, p3_idx):
    """
    Calculates the 2D angle (in degrees) at point p2.
    """
    # 1. Extract ONLY the [x, y] coordinates using [:2], ignoring z
    a = landmarks[p1_idx][:2]
    b = landmarks[p2_idx][:2]
    c = landmarks[p3_idx][:2]

    # 2. Create the vectors BA and BC
    ba = a - b
    bc = c - b

    # 3. Calculate the dot product and the norms (lengths)
    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    # 4. Calculate the cosine of the angle
    cosine_angle = dot_product / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # 5. Calculate the angle in radians and convert it to degrees
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def calc_distance(landmarks, p1_idx, p2_idx):
    """
    Calculates the Euclidean distance between two landmarks.
    """
    # 1. Extract the [x, y] coordinates
    a = landmarks[p1_idx][:2]
    b = landmarks[p2_idx][:2]

    # 2. Calculate the vector between the two points
    vector = a - b

    # 3. Calculate the distance (the magnitude/length of the vector)
    distance = np.linalg.norm(vector)

    return distance

def add_data(normalized_array):
    # Coordonates
    coordonates_array = normalized_array



    # Angles
    # [0] = Elbow
    # [1] = Shoulder
    # [2] = Hip
    # [3] = Knee
    # [4] = Wrist
    # [5] = Ankle
    # [6] = Body - to ground (to knee)
    # [7] = Body - to ground (to hip)
    # [8] = Body - to ground (to shoulder)

    # Left side
    left_angles = [
        calc_angle(normalized_array, L_SHOULDER, L_ELBOW, L_WRIST),
        calc_angle(normalized_array, L_ELBOW, L_SHOULDER, L_HIP),
        calc_angle(normalized_array, L_SHOULDER, L_HIP, L_ANKLE),
        calc_angle(normalized_array, L_HIP, L_KNEE, L_ANKLE),
        calc_angle(normalized_array, L_HAND, L_WRIST, L_ELBOW),
        calc_angle(normalized_array, L_KNEE, L_ANKLE, L_FOOT),
        calc_angle(normalized_array, L_ANKLE, L_KNEE, L_WRIST),
        calc_angle(normalized_array, L_ANKLE, L_HIP, L_WRIST),
        calc_angle(normalized_array, L_ANKLE, L_SHOULDER, L_WRIST),
    ]

    # Right side
    right_angles = [
        calc_angle(normalized_array, R_SHOULDER, R_ELBOW, R_WRIST),
        calc_angle(normalized_array, R_ELBOW, R_SHOULDER, R_HIP),
        calc_angle(normalized_array, R_SHOULDER, R_HIP, R_ANKLE),
        calc_angle(normalized_array, R_HIP, R_KNEE, R_ANKLE),
        calc_angle(normalized_array, R_HAND, R_WRIST, R_ELBOW),
        calc_angle(normalized_array, R_KNEE, R_ANKLE, R_FOOT),
        calc_angle(normalized_array, R_ANKLE, R_KNEE, R_WRIST),
        calc_angle(normalized_array, R_ANKLE, R_HIP, R_WRIST),
        calc_angle(normalized_array, R_ANKLE, R_SHOULDER, R_WRIST),
    ]

    # Concatenate the two arrays. Format final : [[L0, R0], [L1, R1], ... [L8, R8]]
    paired_angles = [[r, l] for r, l in zip(right_angles, left_angles)] # Inverted because image is flipped
    # Convert to a 2D NumPy array for your ML model
    angle_array = np.array(paired_angles, dtype=np.float32)

    # print("Angles : ", angle_array)
    # Display only the elbow angle
    # print(angle_array[0])



    # Distances
    # [0] = Wrist - Shoulder
    # [1] = Ankle - Shoulder

    # Left side
    left_distances = [
        calc_distance(normalized_array, L_WRIST, L_SHOULDER),
        calc_distance(normalized_array, L_ANKLE, L_SHOULDER),
    ]

    # Right side
    right_distances = [
        calc_distance(normalized_array, R_WRIST, R_SHOULDER),
        calc_distance(normalized_array, R_ANKLE, L_SHOULDER),
    ]

    # Concatenate the two arrays. Format final : [[L0, R0], [L1, R1], ... [L8, R8]]
    paired_distances = [[r, l] for r, l in zip(right_distances, left_distances)]  # Inverted because image is flipped
    # Convert to a 2D NumPy array for your ML model
    distance_array = np.array(paired_distances, dtype=np.float32)

    # Display the wrist-shoulder distance
    # print(distance_array[0])


    # Concatenate all of the arrays
    # First transform all of the arrays into a 1D array
    flat_coords = coordonates_array.flatten()
    flat_angles = angle_array.flatten()
    flat_distances = distance_array.flatten()
    final_array = np.concatenate((flat_coords, flat_angles, flat_distances))
    return final_array

def clean_data(pose_array):
    filtered_array = delete_coords(pose_array)
    normalized_array = normalize_coords(filtered_array)
    final_array = add_data(normalized_array)

    # Print each cell one at a time with its index
    # print("\n\n--- Final Feature Vector ---")
    # for index, value in enumerate(final_array):
    #     print(f"[{index}] : {value}")

    return final_array

def extract_features_array():
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
            pose_array = extract_pose_landmarks(body_detected, w, h)

            # 2. Immediately pass it to clean_data()
            final_features = clean_data(pose_array)

            # 3. Print it
            print("Extracted feature array :")
            print(final_features)

    cam.release()
    pose.close()

if __name__ == "__main__":
    extract_features_array()

