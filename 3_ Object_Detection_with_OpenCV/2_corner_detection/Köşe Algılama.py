import cv2
import matplotlib.pyplot as plt
import numpy as np

# Resmi içe aktar

img = cv2.imread("sudoku.jpg", 0)
img = np.float32(img)

print(img.shape)

plt.figure()
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

# Harris Corner Detection
# cv2.cornerHarris(src, blockSize = komşuluk boyutu, ksize, k = free parameter)
dst = cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04)

plt.figure()
plt.imshow(dst, cmap="gray")
plt.axis("off")

# Daha düzgün bir görsel elde etmek için aşağıdaki işlem uygulanır.

dst = cv2.dilate(dst, None)
img[dst > 0.2 * dst.max()] = 1

plt.figure()
plt.imshow(dst, cmap="gray")
plt.axis("off")
plt.show()

# Shi Tomasi Detection

img = cv2.imread("sudoku.jpg", 0)
img = np.float32(img)

corners = cv2.goodFeaturesToTrack(img, maxCorners=120, qualityLevel=0.01, minDistance=10)
corners = np.int64(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 3, (125, 125, 125))

plt.imshow(img)
plt.axis("off")
