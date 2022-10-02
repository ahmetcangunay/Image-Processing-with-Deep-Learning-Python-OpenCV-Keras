# opencv kütüphanesini içe aktaralım
import cv2

# numpy kütüphanesini içe aktaralım
import numpy as np

# resmi siyah beyaz olarak içe aktaralım resmi çizdirelim
img = cv2.imread("odev2.jpg", 0)
# ------cv2.imshow("Image", img)

# resim üzerinde bulunan kenarları tespit edelim ve görselleştirelim
# Edge detection

edges = cv2.Canny(img, 200, 255)
# ------cv2.imshow("Edges", edges)

# yüz tespiti için gerekli haar cascade'i içe aktaralım
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# yüz tespiti yapıp sonuçları görselleştirelim
# (Trackbar ile iyileştirmeler yapalım.)

# Trackbar oluşturma işlemi (Scale ve Neighbors değerleri ayarlayabilmek için.)
cv2.namedWindow("Detection")
cv2.resizeWindow("Detection", width=280, height=360)

# Boş bir fonksiyon(empty) dönebilmek için fonksiyon oluşturuyoruz.


def empty(a):
    pass


# Scale'i değiştirme işlemi
cv2.createTrackbar("Scale", "Detection", 400, 1000, empty)
cv2.createTrackbar("Neighbor", "Detection", 4, 50, empty)


while True:

    img = cv2.imread("odev2.jpg", 0)
    # Detection parameters
    scale_value = 1 + (cv2.getTrackbarPos("Scale", "Detection")/1000)
    neighbor_value = cv2.getTrackbarPos("Neighbor", "Detection")

    print(f"Scale:{scale_value}\nNeighbor:{neighbor_value}\n----------")

    rects = cascade.detectMultiScale(img, scale_value, neighbor_value)

    for (idx1, (x1, y1, w1, h1)) in enumerate(rects):

        cv2.rectangle(img, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)
        cv2.putText(img, f"Yaya {idx1 + 1}", (x1, y1-10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.55, (255, 0, 0), 2)
    cv2.imshow("Detection", img)

    # HOG ilklendirelim insan tespiti algoritmamızı çağıralım ve svm'i
    # set edelim
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects2, weights = hog.detectMultiScale(
        img, padding=(8, 8), scale=scale_value)

    # resme insan tespiti algoritmamızı uygulayalım ve görselleştirelim
    for (idx2, (x2, y2, w2, h2)) in enumerate(rects2):

        cv2.rectangle(img, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 2)
        cv2.putText(img, f"HYaya {idx2 + 1}", (x2, y2-10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.55, (0, 0, 255), 2)
    cv2.imshow("Detection", img)
    if cv2.waitKey(0) & 0xFF == ord("c"):
        continue

    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
