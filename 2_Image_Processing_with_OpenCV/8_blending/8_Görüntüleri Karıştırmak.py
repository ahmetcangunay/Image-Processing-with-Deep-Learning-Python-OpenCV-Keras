import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("img1.jpg")

# Renkleri Convert Etme
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread("img2.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

# Boyutlarını İnceleyelim.
print(img1.shape)
print(img2.shape)

# Resize İşlemi
img1 = cv2.resize(img1, (600, 600))
img2 = cv2.resize(img2, (600, 600))

print(img1.shape)
print(img2.shape)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

# Karıştırılmış Resim = alpha * img1 + beta * img2
# cv2.addWeighted(src1, alpha, src2, beta, gamma = 0)
blended_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
plt.figure()
plt.imshow(blended_img)
