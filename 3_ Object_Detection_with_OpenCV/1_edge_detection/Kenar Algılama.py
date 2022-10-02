# Görüntü parlaklığının keskin bir şekilde değiştiği noktaları tanımlamayı amaçlayan
# bir yöntemdir.

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("london.jpg", 0)

plt.figure()
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

# Herhangi bir threshold değeri vermediğimiz için nehrin üzerindeki girintileri bile aldı!
edges = cv2.Canny(img, threshold1=0, threshold2=255)

plt.figure()
plt.imshow(edges, cmap="gray")
plt.axis("off")
plt.show()

# Resmin medyanını öğrenme ve medyan kullanılarak eşik değerleri oluşturma
median_val = np.median(img)
print(median_val)

# Literatürdeki üst ve alt eşik değerleri elde etme

low = int(max(0, (1 - 0.33) * median_val))
high = int(min(255, (1 + 0.33) * median_val))

print(low)  # Threshold 1
print(high)  # Threshold 2

edges2 = cv2.Canny(img, threshold1=low, threshold2=high)

plt.figure()
plt.imshow(edges2, cmap="gray")
plt.axis("off")
plt.show()

# Tüm resme blurring Uygulama

blurred_img = cv2.blur(img, ksize=(7, 7))

plt.figure()
plt.imshow(blurred_img, cmap="gray")
plt.axis("off")
plt.show()

# Tekrar medyan hesaplama

median_val2 = np.median(blurred_img)
print(median_val2)

low2 = int(max(0, (1 - 0.33) * median_val2))
high2 = int(min(255, (1 + 0.33) * median_val2))

print(low2)  # Threshold 1-2
print(high2)  # Threshold 2-2

edges_blurred = cv2.Canny(blurred_img, threshold1=low2, threshold2=high2)

plt.figure()
plt.imshow(edges_blurred, cmap="gray")
plt.axis("off")
plt.show()
