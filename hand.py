import cv2 as cv
import mediapipe as mp

mp_hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles


def hand_detection():
    # Initialize the Hands model
    hands = mp_hands.Hands(
        static_image_mode=False,  # Set to False for processing video frames
        max_num_hands=2,  # Maximum number of hands to detect
        min_detection_confidence=0.5,  # Minimum confidence threshold for hand detection
        min_tracking_confidence=0.5,
    )

    # Open the camera
    cam = cv.VideoCapture(0)
    
    while True:
                            
        # Read a frame from the camera
        success, frame = cam.read()

        # If the frame is not available, stop the program
        if not success:
            print("Camera frame not available")
            break

        # Convert the frame from BGR to RGB (required by MediaPipe)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process the frame for hand detection and tracking
        hands_detected = hands.process(rgb_frame)

        # If hands are detected, draw landmarks and connections on the frame
        if hands_detected.multi_hand_landmarks:
            for hand_landmarks in hands_detected.multi_hand_landmarks:
                drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )

        # Display every frame, even when no hand is currently detected
        cv.imshow("Show Video", frame)

        # Exit the loop if 'q' key is pressed
        if cv.waitKey(20) & 0xFF == ord("q"):
            break


    hands.close()
    cam.release()
    cv.destroyAllWindows()

hand_detection()