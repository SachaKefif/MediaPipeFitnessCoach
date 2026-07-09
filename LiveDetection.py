import cv2 as cv
import mediapipe as mp
import numpy as np
import xgboost as xgb
from collections import deque
from Data import clean_data, extract_pose_landmarks


def live_detection(frames=60):
    # 1. Load the trained XGBoost model
    print("Loading XGBoost model...")
    model = xgb.XGBClassifier()
    model.load_model("fitness_coach_xgboost.json")
    print("Model loaded successfully!")

    # 2. Initialize MediaPipe and Camera
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    drawing = mp.solutions.drawing_utils
    cam = cv.VideoCapture(1)

    cv.namedWindow("Live Detector", cv.WINDOW_NORMAL)

    # 3. Create the sliding window (conveyor belt)
    window = deque(maxlen=frames)

    print("Starting live detection. Press 'q' to quit.")

    while True:
        success, frame = cam.read()
        if not success:
            break

        h, w, _ = frame.shape
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        body_detected = pose.process(rgb_frame)

        # Draw the skeleton on the frame
        if body_detected.pose_landmarks:
            drawing.draw_landmarks(
                frame,
                body_detected.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Extract data and clean it (get the 1D array of angles/distances)
            pose_array = extract_pose_landmarks(body_detected, w, h)
            features = clean_data(pose_array)

            # Add to our sliding window
            window.append(features)

            # 4. Make a prediction ONLY if the window is full (60 frames captured)
            if len(window) == frames:
                # Flatten the 60 frames into one giant 1D array, then reshape it
                # so XGBoost knows it's looking at 1 single row of data
                xgb_input = np.array(window).flatten().reshape(1, -1)

                # Predict the class (0 or 1)
                prediction = model.predict(xgb_input)[0]

                # Predict the confidence (probability between 0.0 and 1.0)
                # predict_proba returns [[prob_0, prob_1]]
                probability = model.predict_proba(xgb_input)[0][1]

                # 5. Display the result on the screen
                if prediction == 1:
                    # Doing a pushup! (Green text)
                    status_text = f"PUSHUP : ({probability * 100:.1f}%)"
                    color = (0, 255, 0)
                elif prediction == 2:
                    # Doing a squat! (Green text)
                    status_text = f"SQUAT : ({probability * 100:.1f}%)"
                    color = (0, 255, 0)
                else:
                    # Idle / Not doing a pushup (Red text)
                    status_text = f"IDLE : ({probability * 100:.1f}%)"
                    color = (0, 0, 255)

                cv.putText(frame, status_text, (30, 50),
                           cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        else:
            # If no body is detected, clear the window so it doesn't
            # accidentally merge old frames with new frames when walking back in
            window.clear()
            cv.putText(frame, "NO BODY DETECTED", (30, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

        # Show the camera feed
        cv.imshow("Live Pushup Detector", frame)

        # Quit if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    pose.close()
    cv.destroyAllWindows()


if __name__ == "__main__":
    live_detection(frames=60)
