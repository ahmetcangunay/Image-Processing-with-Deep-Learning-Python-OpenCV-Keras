import cv2
import numpy as np

img = cv2.imread("lenna.png")
cv2.imshow("Original", img)

# Horizontal (Yatay) Birleştirme
hor = np.hstack((img, img))  # img bir arraydir.
cv2.imshow("Horizontal", hor)

# Vertical (Dikey) Birleştirme
ver = np.vstack((img, img))
cv2.imshow("Vertical", ver)
