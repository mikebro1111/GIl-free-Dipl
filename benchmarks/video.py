import time
import cv2

class VideoProcessing:
    def __init__(self, video_file):
        self.video_file = video_file

    def process_video(self):
        cap = cv2.VideoCapture(self.video_file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        cap.release()
        end = time.time()

        return end - start
    
def benchmark_video_processing(video_file):
    video_processor = VideoProcessing(video_file)
    return video_processor.process_video()
