import cv2
import matplotlib.pyplot as plt

# Resmi İçe Aktar
img = cv2.imread("img1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure()
# ColorMap Cmap = "gray" kısmı resmi siyah-beyaza çevirir.
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

# Eşikleme İşlemi
# cv2.threshold(src = img, thresh = eşik değeri, maxval = en yüksek değer, type = binary or inverse)
# Binary eşik değeri ile max arasını beyazlar. Kalan kısımlar siyahtır. Inverse tam tersi.
_, threshed_img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)

plt.figure()
plt.imshow(threshed_img, cmap="gray")
plt.axis("off")
plt.show()

# Bütünlüğü bozmamak için "Adaptive Threshold" kullanacağız.
# Örnek olarak dağın aydınlık kısmı ile karanlık kısmı verilebilir.
# cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize = Komşu Sayısı, C)
threshed_img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

plt.figure()
plt.imshow(threshed_img2, cmap="gray")
plt.axis("off")
plt.show()
