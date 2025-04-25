import cv2
import time
import os

def process_video(source=0, save=False, output_path="results/processed_video.avi"):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Failed to open video source")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    if save:
        os.makedirs("results", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    frame_count = 0
    total_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        end = time.time()

        total_time += (end - start)
        frame_count += 1

        if save:
            out.write(gray)

        cv2.imshow('Grayscale Video', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save:
        out.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames in {total_time:.2f} seconds ({frame_count / total_time:.2f} FPS)")
