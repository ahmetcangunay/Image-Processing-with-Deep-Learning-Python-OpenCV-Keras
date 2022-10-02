import cv2
import numpy as np

img = cv2.imread("kart.png")
cv2.imshow("Original", img)

width = 400
height = 500

# Kartın Şuanki Köşe Değerleri
pts1 = np.float32([[203, 1], [1, 472], [540, 150], [338, 617]])

# Kartın Yeni Köşe Değerleri
pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])

# Perspektif Transform Matrisi Elde Etme
matrix = cv2.getPerspectiveTransform(pts1, pts2)
print(matrix)

# Çevirme İşlemi Gerçekleştirme (Resim, matrix, (genişlik, yükseklik))
imgOutput = cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow("Cevirilmis", imgOutput)
