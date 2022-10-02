import cv2
import matplotlib.pyplot as plt
import numpy as np

coin = cv2.imread("coins.jpg")

plt.figure()
plt.imshow(coin)
plt.axis("off")
plt.show()

# LPF Bluring (Median Blur)
coin_blur = cv2.medianBlur(coin, 13)

plt.figure()
plt.imshow(coin_blur)
plt.axis("off")
plt.show()

# Gray Scale
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(coin_gray, cmap="gray")
plt.axis("off")
plt.show()

# Binary Threshold
ret, coin_thresh = cv2.threshold(coin_gray, 75, 255, cv2.THRESH_BINARY)

plt.figure()
plt.imshow(coin_thresh, cmap="gray")
plt.axis("off")
plt.show()

# Kontur Bulma İşlemi
contours, hierarchy = cv2.findContours(
    coin_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:  # External contour
        cv2.drawContours(coin, contours, i, (0, 255, 0), 10)

plt.figure()
plt.imshow(coin)
plt.axis("off")
plt.show()

# Segmentasyonu tam olarak gerçekleştiremedik!

# %% Watershed (Havza)
coin = cv2.imread("coins.jpg")

plt.figure()
plt.imshow(coin)
plt.axis("off")
plt.show()

# LPF Bluring (Median Blur)
coin_blur = cv2.medianBlur(coin, 13)

plt.figure()
plt.imshow(coin_blur)
plt.axis("off")
plt.show()

# Gray Scale
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(coin_gray, cmap="gray")
plt.axis("off")
plt.show()

# Binary Threshold
ret, coin_thresh = cv2.threshold(coin_gray, 65, 255, cv2.THRESH_BINARY)

plt.figure()
plt.imshow(coin_thresh, cmap="gray")
plt.axis("off")
plt.show()

# Açılma işlemi ile paralar arasındaki boşlukları arttırıyoruz.
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(coin_thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Nesneler arası distance bulma işlemi
# cv2.DIST_L2 (Öklid)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

plt.figure()
plt.imshow(dist_transform, cmap="gray")
plt.axis("off")
plt.show()

# Öndeki resmi bulabilmek için resmi küçültelim.
ret, sure_foreground = cv2.threshold(
    dist_transform, 0.4*np.max(dist_transform), 255, 0)

plt.figure()
plt.imshow(sure_foreground, cmap="gray")
plt.axis("off")
plt.show()

# Aradaki köprüleri kırmış olduk. (Genlik değeri az olan köprüleri
# threshold metodu ile yıkarak adacıklar elde ettik.)

# Arka plan için resmi büyütme işlemi
sure_background = cv2.dilate(opening, kernel, iterations=1)
sure_foreground = np.uint8(sure_foreground)

unknown = cv2.subtract(sure_background, sure_foreground)

plt.figure()
plt.imshow(unknown, cmap="gray")
plt.axis("off")
plt.show()

# Bağlantı
ret, marker = cv2.connectedComponents(sure_foreground)
marker += 1
marker[unknown == 255] = 0

plt.figure()
plt.imshow(marker, cmap="gray")
plt.axis("off")
plt.show()

# Gerçek Havza Algoritması :)
marker = cv2.watershed(coin, marker)

plt.figure()
plt.imshow(marker, cmap="gray")
plt.axis("off")
plt.show()

# Kontur Bulma İşlemi (Yeniden)
contours, hierarchy = cv2.findContours(
    marker.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:  # External contour
        cv2.drawContours(coin, contours, i, (255, 0, 0), 3)

plt.figure()
plt.imshow(coin, cmap="gray")
plt.axis("off")
plt.show()
