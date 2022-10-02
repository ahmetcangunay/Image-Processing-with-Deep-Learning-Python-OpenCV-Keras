# %% Gradyanlar

# Görüntü gradyanı, görüntüdeki yoğunluk veya renkteki yönlü bir değişikliktir.
# Kenar algılamada kullanılır.

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("sudoku.jpg", 0)

plt.figure()
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("Original")
plt.show()

# x gradyanları
# cv2.Sobel(src, ddepth = output derinliği, dx = x yönü, dy = y yönü, ksize = kernel size)
sobelx = cv2.Sobel(img, cv2.CV_16S, 1, 0, 5)

plt.figure()
plt.imshow(sobelx, cmap="gray")
plt.axis("off")
plt.title("Sobel X")
plt.show()

# y gradyanları
# cv2.Sobel(src, ddepth = output derinliği, dx = x yönü, dy = y yönü, ksize = kernel size)
sobely = cv2.Sobel(img, cv2.CV_16S, 0, 1, 5)

plt.figure()
plt.imshow(sobely, cmap="gray")
plt.axis("off")
plt.title("Sobel Y")
plt.show()

# Laplacian gradyan
# cv2.Laplacian(src, ddepth = output derinliği)
laplacian = cv2.Laplacian(img, cv2.CV_16S)

plt.figure()
plt.imshow(laplacian, cmap="gray")
plt.axis("off")
plt.title("Laplacian")
plt.show()
