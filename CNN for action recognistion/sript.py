import cv2

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()

        print(ret, frame, video_path)
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    print(f"len(frames): {len(frames)}\n")
    frames = np.array(frames)
    print(f"frames.shape: {frames.shape}\n")
    return frames

load_video_frames("data/mini_UCF/BabyCrawling/v_BabyCrawling_g25_c06.avi")