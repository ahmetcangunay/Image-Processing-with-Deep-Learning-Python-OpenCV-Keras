import cv2

# RESMİ İÇERİ AKTARMA
# Resim ve Python dosyasının aynı klasörde olması kolaylık sağlar.

img = cv2.imread("messi5.jpg", 0)  # 0, resmi gray scale olarak aktarmamızı sağlar.

# Görselleştirme

cv2.imshow("ilk Resim", img)

# Klavye komutu oluşturma

k = cv2.waitKey(0) & 0xFF

if k == 27:  # esc'nin unicode karşılığı
    cv2.destroyAllWindows()
elif k == ord('s'):  # 's' harfinin Unicode karşılığı
    cv2.imwrite("messi_gray.png", img)  # Dosya kaydetme
    cv2.destroyAllWindows()
