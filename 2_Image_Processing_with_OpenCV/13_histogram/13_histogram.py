# Görüntü histogramı, dijital görüntüdeki ton dağılımının grafiksel bir temsili
# olarak işlev gören bir histogram türüdür.

# Her bir ton değeri için piksel sayısını içerir.

# Belirli bir görüntü için histograma bakılarak, ton dağılımı anlaşılabilir.

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Resmi içe aktarma

img = cv2.imread("red_blue.jpg")
img_vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img_vis)
plt.axis("off")
plt.show()

print(img.shape)

# Histogram oluşturma
# cv2.calcHist(images, channels = gray or RGB, mask, histSize, ranges)
img_hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

print(img_hist.shape)

plt.figure()
plt.plot(img_hist)
plt.show()

color = ("b", "g", "r")

plt.figure()
for i, c in enumerate(color):
    hist = cv2.calcHist([img], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
    plt.plot(hist, color=c)

# Golden Gate

golden_gate = cv2.imread("goldenGate.jpg")
golden_gate_vis = cv2.cvtColor(golden_gate, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(golden_gate_vis)
plt.show()

print(golden_gate.shape)

# Maske oluşturma (Bütün resmi kullanmak işlerimizi zorlaştıracaktır.)

mask = np.zeros(golden_gate.shape[:2], np.uint8)

plt.figure()
plt.imshow(mask, cmap="gray")
plt.show()

# Maskede Delik Açma İşlemi

mask[1500:2000, 1000:2000] = 255

plt.figure()
plt.imshow(mask, cmap="gray")
plt.show()

# Maskeyi Resme Uygulama

masked_img_vis = cv2.bitwise_and(src1=golden_gate_vis, src2=golden_gate_vis, mask=mask)

plt.figure()
plt.imshow(masked_img_vis, cmap="gray")
plt.show()

masked_img = cv2.bitwise_and(src1=golden_gate, src2=golden_gate, mask=mask)
masked_img_hist = cv2.calcHist([golden_gate], channels=[0], mask=mask, histSize=[256], ranges=[0, 256])

plt.figure()
plt.plot(masked_img_hist)
plt.show()

# Histogram Eşitleme
# Kontrastı arttırır.

img = cv2.imread("hist_equ.jpg", 0)

plt.figure()
plt.imshow(img, cmap="gray")
plt.show()

img_hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

plt.figure()
plt.plot(img_hist)
plt.show()

eq_hist = cv2.equalizeHist(img)

plt.figure()
plt.imshow(eq_hist, cmap="gray")
plt.show()

eq_img_hist = cv2.calcHist([eq_hist], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

plt.figure()
plt.plot(eq_img_hist)
plt.show()
