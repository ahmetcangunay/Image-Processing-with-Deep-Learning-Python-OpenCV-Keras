import cv2
import matplotlib.pyplot as plt
import numpy as np

# resmi içe aktarma
img = cv2.imread("datai_team.jpg", 0)

plt.figure()
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("Original")
plt.show()

# %% Erozyon (Sınırları Küçültme İşlemi)

kernel = np.ones((5, 5), dtype=np.uint8)
# cv2.erode(src, kernel,iteration)
result = cv2.erode(img, kernel, iterations=1)

plt.figure()
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.title("Erode")
plt.show()

# %% Genişleme (Dilation) Erozyonun tam tersi

result2 = cv2.dilate(img, kernel, iterations=1)

plt.figure()
plt.imshow(result2, cmap="gray")
plt.axis("off")
plt.title("Dilation")
plt.show()

# %% Açılma (Beyaz Gürültüyü Azaltmak için) Önce daraltma sonra genişletme

# Öncesinde Sistemimize Beyaz bir gürültü oluşturalım.

whiteNoise = np.random.randint(0, 2, size=img.shape[:2])
whiteNoise *= 255

plt.figure()
plt.imshow(whiteNoise, cmap="gray")
plt.axis("off")
plt.title("White Noise")
plt.show()

white_noisy_img = img + whiteNoise

plt.figure()
plt.imshow(white_noisy_img, cmap="gray")
plt.axis("off")
plt.title("Noisy Image")
plt.show()

# Açılma Yöntemi Uygulanışı
opening = cv2.morphologyEx(white_noisy_img.astype(np.float32), cv2.MORPH_OPEN, kernel)

plt.figure()
plt.imshow(opening, cmap="gray")
plt.axis("off")
plt.title("Opened Image")
plt.show()

# %% Kapatma (Siyah Gürültüyü Azaltmak için) Önce genişletme sonra daraltma

# Öncesinde Sistemimize Siyah bir gürültü oluşturalım.

blackNoise = np.random.randint(0, 2, size=img.shape[:2])
blackNoise *= -255

plt.figure()
plt.imshow(blackNoise, cmap="gray")
plt.axis("off")
plt.title("Black Noise")
plt.show()

black_noisy_img = blackNoise + img

# Filtreleme İşlemi
black_noisy_img[black_noisy_img <= -245] = 0

plt.figure()
plt.imshow(black_noisy_img, cmap="gray")
plt.axis("off")
plt.title("Black Noisy Image")
plt.show()

# Kapatma Yöntemi Uygulanışı
closing = cv2.morphologyEx(black_noisy_img.astype(np.float32), cv2.MORPH_CLOSE, kernel)

plt.figure()
plt.imshow(closing, cmap="gray")
plt.axis("off")
plt.title("Closed Image")
plt.show()

# %% Gradient -> Dilation ile Erose arasındaki farkı alır.

# Kenar tespitinde kullanılan yöntemlerden biridir.

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

plt.figure()
plt.imshow(gradient, cmap="gray")
plt.axis("off")
plt.title("Gradient Image")
plt.show()
