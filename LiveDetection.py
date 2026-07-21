import cv2 as cv
import mediapipe as mp
import numpy as np
import xgboost as xgb
import tensorflow as tf
import winsound
from collections import deque
from Data import clean_data, extract_pose_landmarks

def play_detected_cue():
    # Plays a lower-pitched beep (500 Hz for 300 ms)
    winsound.Beep(1750, 300)

def live_detection(frames=60):
    # 1. Load the trained model

    # print("Loading XGBoost model...")
    # model = xgb.XGBClassifier()
    # model.load_model("fitness_coach_xgboost.json")

    print("Loading Neural Network model...")
    model = tf.keras.models.load_model("fitness_coach_lstm.keras")

    print("Model loaded successfully!")

    # 2. Initialize MediaPipe and Camera
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    drawing = mp.solutions.drawing_utils

    # phone_ip = "192.168.1.27"
    # cam = cv.VideoCapture(f"http://{phone_ip}:4747/video")
    cam = cv.VideoCapture(0)

    print("Camera Opened:", cam.isOpened())
    if not cam.isOpened():
        print("Failed to open camera")
        return

    cv.namedWindow("Live Detector", cv.WINDOW_NORMAL)
    cv.resizeWindow("Live Detector", 1440, 810)

    # 3. Create the sliding window (conveyor belt)
    window = deque(maxlen=frames)

    # Variables to count exercise repetitions
    tot_pushup = 0
    tot_squat = 0
    tot_crunch = 0

    # Prevent multiple detections of the same repetition
    ready_for_new_rep = True

    print("Starting live detection. Press 'q' to quit.")

    while True:
        success, frame = cam.read()
        if not success:
            break

        # Flip the frame
        frame = cv.flip(frame, 1)

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
                # xgb_input = np.array(window).flatten().reshape(1, -1)
                # # Predict the class (0 or 1)
                # prediction = model.predict(xgb_input)[0]
                #
                # # Predict the confidence (probability between 0.0 and 1.0)
                # all_probabilities = model.predict_proba(xgb_input)[0]
                # probability = all_probabilities[prediction]

                # Calculate how many features are in a single frame
                features_per_frame = len(window[0])
                # Reshape the 60 frames into a 3D block: [1 sample, 60 frames, features]
                # so the LSTM knows it is looking at a sequence over time
                nn_input = np.array(window).reshape(1, frames, features_per_frame)
                #
                # Get the list of probabilities for all 4 classes [prob_0, prob_1, prob_2, prob_3]
                # verbose=0 stops Keras from spamming your console with prediction logs
                all_probabilities = model.predict(nn_input, verbose=0)[0]
                # Grab the class index (0, 1, 2, or 3) with the highest probability
                prediction = np.argmax(all_probabilities)
                # Grab that specific confidence percentage
                probability = all_probabilities[prediction]

                # 5. Display the result on the screen
                # Default display
                status_text = f"Label {prediction}"
                color = (0, 255, 0)

                # -----------------------------
                # IDLE
                # -----------------------------
                if prediction == 0:
                    ready_for_new_rep = True

                    status_text = f"IDLE ({probability * 100:.1f}%)"
                    color = (0, 0, 255)

                # -----------------------------
                # PUSHUP
                # -----------------------------
                elif prediction == 1:
                    status_text = f"PUSHUP ({probability * 100:.1f}%) | Total: {tot_pushup}"

                    if ready_for_new_rep:
                        play_detected_cue()
                        tot_pushup += 1
                        ready_for_new_rep = False

                # -----------------------------
                # SQUAT
                # -----------------------------
                elif prediction == 2:
                    status_text = f"SQUAT ({probability * 100:.1f}%) | Total: {tot_squat}"

                    if ready_for_new_rep:
                        play_detected_cue()
                        tot_squat += 1
                        ready_for_new_rep = False

                # -----------------------------
                # CRUNCH
                # -----------------------------
                elif prediction == 3:
                    status_text = f"CRUNCH ({probability * 100:.1f}%) | Total: {tot_crunch}"

                    if ready_for_new_rep:
                        play_detected_cue()
                        tot_crunch += 1
                        ready_for_new_rep = False

                cv.putText(frame, status_text, (30, 50),
                           cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        else:
            # If no body is detected, clear the window so it doesn't
            # accidentally merge old frames with new frames when walking back in
            window.clear()
            cv.putText(frame, "NO BODY DETECTED", (30, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

        # Show the camera feed
        display_frame = cv.resize(frame, (1440, 810))
        cv.imshow("Live Detector", display_frame)

        # Quit if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    pose.close()
    cv.destroyAllWindows()


if __name__ == "__main__":
    live_detection(frames=60)
