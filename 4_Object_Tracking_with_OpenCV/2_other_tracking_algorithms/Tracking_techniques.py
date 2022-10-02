import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

OPENCV_OBJECT_TRACKERS = {"csrt": cv2.legacy.TrackerCSRT_create,
                          "kcf": cv2.legacy.TrackerKCF_create,
                          "boosting": cv2.legacy.TrackerBoosting_create,
                          "mil": cv2.legacy.TrackerMIL_create,
                          "tld": cv2.legacy.TrackerTLD_create,
                          "medianflow": cv2.legacy.TrackerMedianFlow_create,
                          "mosse": cv2.legacy.TrackerMOSSE_create
                          }


tracker_name = "boosting"

# Burada fonksiyon döneceğinden parantez koymak gereklidir.
tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()

print("Tracker:", tracker_name)

# Gt yükleme işlemi
gt = pd.read_csv("gt_new.txt")


video_path = "MOT17-13-SDP.mp4"

cap = cv2.VideoCapture(video_path)

# Gerekli parametreler

initBB = None
fps = 25
frame_number = []
f = 0
success_track_frame_success = 0
track_list = []
start_time = time.time()


while True:
    time.sleep(1/fps)

    ret, frame = cap.read()

    if ret:

        frame = cv2.resize(frame, dsize=(960, 540))
        (H, W) = frame.shape[:2]

        # gt
        car_gt = gt[gt.frame_no == f]

        if len(car_gt) != 0:

            x = car_gt.x.values[0]
            y = car_gt.y.values[0]
            w = car_gt.w.values[0]
            h = car_gt.h.values[0]

            center_x = car_gt.center_x.values[0]
            center_y = car_gt.center_y.values[0]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), -1)

        #
        cv2.imshow("Tracking", frame)
        # key
        key = cv2.waitKey(1) & 0xFF

        if key == ord('t'):
            cv2.selectROI("Frame", frame, fromCenter=False)
        elif key == ord('q'):
            break

        # frame
        frame_number.append(f)
        f += 1
    else:
        break

cap.release()
cv2.destroyAllWindows()
