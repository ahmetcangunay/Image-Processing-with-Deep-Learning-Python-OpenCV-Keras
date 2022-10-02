# opencv kütüphanesini içe aktaralım
import cv2

# matplotlib kütüphanesini içe aktaralım
import matplotlib.pyplot as plt

# resmi siyah beyaz olarak içe aktaralım
img = cv2.imread("odev1.jpg", 0)
# resmi çizdirelim
cv2.imshow("IMG", img)

# resmin boyutuna bakalım
print(img.shape)

# resmi 4/5 oranında yeniden boyutlandıralım ve resmi çizdirelim
ratio = 4 / 5
resized_img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))

print(resized_img.shape)

plt.figure()
plt.imshow(resized_img, cmap="gray")
plt.axis("off")
plt.show()

# orijinal resme bir yazı ekleyelim mesela "kopek" ve resmi çizdirelim
img_text = cv2.putText(img, "Kopek", (350, 350), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0))

plt.figure()
plt.imshow(img_text, cmap="gray")
plt.axis("off")
plt.show()

# orijinal resmin 50 threshold değeri üzerindekileri beyaz yap altındakileri siyah yapalım, 
# binary threshold yöntemi kullanalım ve resmi çizdirelim
_, threshed_img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

plt.figure()
plt.imshow(threshed_img, cmap="gray")
plt.axis("off")
plt.show()

# orijinal resme gaussian bulanıklaştırma uygulayalım ve resmi çizdirelim
gb = cv2.GaussianBlur(img, (3, 3), 7)
plt.figure()
plt.imshow(gb, cmap="gray")
plt.axis("off")
plt.show()
# orijinal resme Laplacian  gradyan uygulayalım ve resmi çizdirelim
laplacian = cv2.Laplacian(img, cv2.CV_64F)

cv2.imshow("Laplacian", laplacian)

# orijinal resmin histogramını çizdirelim
img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.figure()
plt.plot(img_hist)
plt.show()
