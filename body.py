import cv2 as cv
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def main():
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cam = cv.VideoCapture(0)

    if not cam.isOpened():
        print("Camera not found")
        return

    window_name = "Pose Detection"
    cv.namedWindow(window_name)

    while True:
        success, frame = cam.read()
        if not success:
            break

        frame = cv.flip(frame, 1)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_styles.get_default_pose_landmarks_style()
            )

        cv.imshow(window_name, frame)

        # quit on X or Q
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

        if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()