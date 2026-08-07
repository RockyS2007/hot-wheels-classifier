import cv2

video = cv2.VideoCapture('raw_video/grey_lambo_3.mov')    # Video Location Here

sample_fps = 2
video_fps = video.get(cv2.CAP_PROP_FPS)
sample_interval = max(1, round(video_fps / sample_fps))

frame_number = 0
saved_number = 0

while True:
    success, frame = video.read()

    if not success:
        break

    if frame_number % sample_interval == 0:
        cv2.imwrite(f'./data/train/lamborghini/grey_lambo_3_{saved_number:05d}.jpg', frame)  # write image to path
        saved_number += 1   
    frame_number += 1

video.release()
# cv2.destroyAllWindows()
