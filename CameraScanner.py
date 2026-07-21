import cv2 as cv

print("Scanning camera indices...")
for index in range(5):
    cap = cv.VideoCapture(index, cv.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✅ Camera Index {index} is WORKING! Frame size: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"❌ Camera Index {index} opened but returned no video.")
        cap.release()
    else:
        print(f"❌ Camera Index {index} could not be opened.")