import cv2

img = cv2.imread("lenna.png")

print("Resim Boyutu", img.shape)

cv2.imshow("Lenna", img)

# Yeniden Boyutlandırma
imgResized = cv2.resize(img, (800, 800))
print("Yeni Boyut:", imgResized.shape)
cv2.imshow("Lena_New", imgResized)

# Crop (Kırpma) İşlemi

imgCropped = img[:200, :300]
cv2.imshow("Cropped", imgCropped)
