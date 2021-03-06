# -*- coding: utf-8 -*-
from imutils.video import FileVideoStream
from imutils.video import FPS
import imutils
import time
from detect_lane import *

cap = cv2.VideoCapture("test_videos/test2.mp4")
fps = FPS().start()
_, frame = cap.read()
# frame = imutils.resize(frame, width=450)
imshape = frame.shape
A = (0.42*imshape[1], 0.53*imshape[0])
B = (0.58*imshape[1], 0.53*imshape[0])
C = (0.82*imshape[1], 0.75*imshape[0])
D = (0.25*imshape[1], 0.75*imshape[0])
vertices = np.array([[B, C, D, A]])


def process_video(video_input, video_output):
    detector = LaneDetector()

    clip = VideoFileClip(os.path.join('test_videos', video_input))
    processed = clip.fl_image(detector.process)
    processed.write_videofile(os.path.join(
        'output_videos', video_output), audio=False)


if __name__ == '__main__':
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=450)
        detector = LaneDetector()
        processed_img = detector.process(frame)
        # processed_img = region_of_interest(processed_img, vertices)

        cv2.imshow("Screen", processed_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cap.release()
    cv2.destroyAllWindows()
