import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from collections import deque
from Data import clean_data, extract_pose_landmarks


def record_action_with_pauses(label, frames=60):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cam = cv.VideoCapture(1)

    cv.namedWindow("Recording Station")
    dataset = []

    print(f"Starting recording session for label: {label}")

    sample_number = 1

    while True:
        # ---------------------------------------------------------
        # PHASE 1: PREPARATION (3 seconds)
        # ---------------------------------------------------------
        start_time = time.time()
        while time.time() - start_time < 3.0:
            success, frame = cam.read()
            if not success: break

            # If the label is not 0 (do nothing)
            if label != 0:
                # Show yellow warning text
                cv.putText(frame, f"Sample {sample_number}: GET READY.... {(time.time() - start_time)}s remaining",
                           (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv.imshow("Recording Station", frame)
            else:
                # Show yellow warning text
                cv.putText(frame, f"Sample {sample_number}: QUIT BY PRESSING Q...",
                           (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv.imshow("Recording Station", frame)

            # Allow window to update and check for 'q' to quit early
            if cv.waitKey(1) & 0xFF == ord('q'):
                # Clean up and Save
                cam.release()
                pose.close()
                cv.destroyAllWindows()

                # Save to CSV (append mode)
                df = pd.DataFrame(dataset)

                filename = f"dataset_label_{label}.csv"

                df.to_csv(
                    filename,
                    mode="a",  # append instead of overwrite
                    index=False,
                    header=False
                )

                print(f"\nSuccessfully appended {sample_number - 1} samples to {filename}")
                return

        # ---------------------------------------------------------
        # PHASE 2: RECORDING (X Frames)
        # ---------------------------------------------------------
        window = deque(maxlen=frames)
        while len(window) < frames:
            success, frame = cam.read()
            if not success: break

            h, w, _ = frame.shape
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            body_detected = pose.process(rgb_frame)

            # Draw the pose landmarks on the recording frame
            if body_detected.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    body_detected.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 0, 255),  # Red joints
                        thickness=3,
                        circle_radius=3
                    ),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(255, 255, 255),  # White bones
                        thickness=2
                    )
                )

            # Show red recording text
            cv.putText(frame, f"RECORDING: {len(window)}/{frames}",
                       (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv.imshow("Recording Station", frame)
            cv.waitKey(1)

            if body_detected.pose_landmarks:
                pose_array = extract_pose_landmarks(body_detected, w, h)
                features = clean_data(pose_array)
                window.append(features)

        # Save the X-frame window as one sample
        flattened_window = np.array(window).flatten()
        labeled_data = np.append(flattened_window, label)
        dataset.append(labeled_data)
        print(f"Saved sample {sample_number}")

        # ---------------------------------------------------------
        # PHASE 3: RELAX (3 second)
        # ---------------------------------------------------------
        # Skip if the goal is to do nothing, to same time
        if label != 0:
            start_time = time.time()
            while time.time() - start_time < 3.0:
                success, frame = cam.read()
                if not success: break

                # Show green relax text
                cv.putText(frame, "RELAX", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.imshow("Recording Station", frame)
                cv.waitKey(1)

        sample_number += 1


if __name__ == "__main__":
    # Record X distinct samples. Change label to 0, 1, 2 etc. for different moves.
    record_action_with_pauses(label=2, frames=60)
