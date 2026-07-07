import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from collections import deque
from Data import clean_data
import Visualization


def record_action_with_pauses(label, num_samples=30, frames=60):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cam = cv.VideoCapture(1)

    cv.namedWindow("Recording Station")
    dataset = []

    print(f"Starting recording session for label: {label}")

    for i in range(num_samples):
        # ---------------------------------------------------------
        # PHASE 1: PREPARATION (2 seconds)
        # ---------------------------------------------------------
        start_time = time.time()
        while time.time() - start_time < 2.0:
            success, frame = cam.read()
            if not success: break

            # Show yellow warning text
            cv.putText(frame, f"Sample {i + 1}/{num_samples}: GET READY...",
                       (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv.imshow("Recording Station", frame)

            # Allow window to update and check for 'q' to quit early
            if cv.waitKey(1) & 0xFF == ord('q'):
                cam.release()
                cv.destroyAllWindows()
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

            # Show red recording text
            cv.putText(frame, f"RECORDING: {len(window)}/{frames}",
                       (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv.imshow("Recording Station", frame)
            cv.waitKey(1)

            if body_detected.pose_landmarks:
                pose_array = Visualization.extract_pose_landmarks(body_detected, w, h)
                features = clean_data(pose_array)
                window.append(features)

        # Save the X-frame window as one sample
        flattened_window = np.array(window).flatten()
        labeled_data = np.append(flattened_window, label)
        dataset.append(labeled_data)
        print(f"Saved sample {i + 1}/{num_samples}")

        # ---------------------------------------------------------
        # PHASE 3: RELAX (1 second)
        # ---------------------------------------------------------
        start_time = time.time()
        while time.time() - start_time < 1.0:
            success, frame = cam.read()
            if not success: break

            # Show green relax text
            cv.putText(frame, "RELAX", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.imshow("Recording Station", frame)
            cv.waitKey(1)

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

    print(f"\nSuccessfully appended {num_samples} samples to {filename}")


if __name__ == "__main__":
    # Record 30 distinct samples. Change label to 0, 1, 2 etc. for different moves.
    record_action_with_pauses(label=1, num_samples=30)