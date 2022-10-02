import cv2
import numpy as np

# Resim Oluştur
img = np.zeros((512, 512, 3), np.uint8)  # Siyah bir resim
print(img.shape)

cv2.imshow("Siyah", img)

# Line
# cv2.line(Resim, başlangıç noktası,bitiş noktası,renk,kalınlık)
cv2.line(img, (0, 0), (512, 512), (0, 255, 0), 3)  # BGR(0,255,0) -> Green
cv2.imshow("Line", img)  # Tekrar Görüntüle

# Rectangle (Dikdörtgen)
# cv2.rectangle(Resim, başlangıç noktası,bitiş noktası,renk,kalınlık)

cv2.rectangle(img, (0, 0), (256, 256), (255, 0, 0), cv2.FILLED)  # içini doldurma işlemi için "filled"
cv2.imshow("Rectangle", img)

# Circle (Çember)
# cv2.circle(Resim, merkez noktası, yarıçap, renk, kalınlık)

cv2.circle(img, (300, 300), 45, (0, 0, 255), cv2.FILLED)
cv2.imshow("Circle", img)

# Text
# cv2.putText(Resim, yazı, başlangıç noktası, yazı fontu, font kalınlığı, renk)
cv2.putText(img, "Resim", (350, 350), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
cv2.imshow("Text", img)
