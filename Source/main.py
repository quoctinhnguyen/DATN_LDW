# -*- coding: utf-8 -*-
from imutils.video import FPS
import imutils
from detect_lane import *

<<<<<<< HEAD
cap = cv2.VideoCapture("test_videos/test2.mp4")
fps = FPS().start()
=======
cap = cv2.VideoCapture("test_videos/test2.flv")
>>>>>>> 21eef618c3615d1c6daf47804e299b0f714838ba
_, frame = cap.read()
imshape = frame.shape
A = (0.42*imshape[1], 0.53*imshape[0])
B = (0.58*imshape[1], 0.53*imshape[0])
C = (0.82*imshape[1], 0.75*imshape[0])
D = (0.25*imshape[1], 0.75*imshape[0])
vertices = np.array([[B, C, D, A]])

# cap.set(cv2.CAP_PROP_FPS, 10)

# # Find OpenCV version
# (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# if int(major_ver) < 3:
#     fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
#     print(
#         "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
# else:
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(
#         "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


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
        # processed_img = preprocess_image(frame)
        # detector = LaneDetector()
        # processed_img = detector.process(frame, vertices)
        # processed_img = region_of_interest(processed_img, vertices)
        # resize the frame and convert it to grayscale (while still
        # retaining 3 channels)
        frame = imutils.resize(frame, width=450)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.dstack([frame, frame, frame])

        # display a piece of text to the frame (so we can benchmark
        # fairly against the fast method)
        cv2.putText(frame, "Slow Method", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Screen", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cap.release()
    cv2.destroyAllWindows()
