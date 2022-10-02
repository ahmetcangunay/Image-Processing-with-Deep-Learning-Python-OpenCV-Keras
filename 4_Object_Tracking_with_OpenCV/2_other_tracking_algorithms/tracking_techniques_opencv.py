import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

OPENCV_OBJECT_TRACKERS = {"csrt": cv2.TrackerCSRT_create,
                          "kcf": cv2.TrackerKCF_create,
                          "boosting": cv2.legacy.TrackerBoosting_create,
                          "mil": cv2.TrackerMIL_create,
                          "tld": cv2.TrackerMIL_create,
                          "medianflow": cv2.legacy.TrackerMedianFlow_create,
                          "mosse": cv2.legacy.TrackerMOSSE_create}

tracker_name = "boosting"
tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
print("Tracker:", tracker_name)

gt = pd.read_csv("gt_new.txt")

video_path = "MOT17-13-SDP.mp4"

cap = cv2.VideoCapture(video_path)
