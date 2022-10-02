import cv2
import numpy as np
from collections import deque
from time import sleep

# Nesne merkezini depolayacak veri tipi
buffer_size = 16  # deque boyutu
pts = deque(maxlen=buffer_size)  # merkez noktaları

# Mavi renk aralığı belirleme (HSV formatında)

# HSV -> H(Hue - Ton)/ S(Saturation - Doygunluk)/ V(Value - Değer)
blue_lower = (84, 98, 0)
blue_upper = (179, 255, 255)

# Capture
cap = cv2.VideoCapture(0)

cap.set(propId=3, value=960)
cap.set(propId=4, value=480)

while True:

    success, img_original = cap.read()

    if success:

        # Blur
        # sigmaX = Standart Deviation
        blurred = cv2.GaussianBlur(img_original, ksize=(11, 11), sigmaX=0)

        # HSV renk skalasına geçiş
        hsv = cv2.cvtColor(blurred, code=cv2.COLOR_BGR2HSV)

        cv2.imshow("HSV Image", hsv)

        # Mavi renk için maske oluşturma
        mask = cv2.inRange(src=hsv, lowerb=blue_lower, upperb=blue_upper)
        # -----------cv2.imshow("Mask Image", mask)

        # Maskenin etrafında oluşan gürültüleri silme işlemi
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # -----------cv2.imshow("Mask + Erode + Dilation", mask)

        # Kontur işlemi
        (_, contours, _) = cv2.findContours(
            image=mask.copy(), mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE)
        center = None  # Nesnenin merkezi olacak.

        if len(contours) > 0:  # Bir kontur bulabilmişsek bu kısım çalışacak.

            # En büyük konturu alma işlemi
            c = max(contours, key=cv2.contourArea)

            # Dikdörtgene çevirme işlemi(Konturu kapsayacak min alanlı rect)
            rect = cv2.minAreaRect(c)

            ((x, y), (width, height), rotation) = rect
            s = "x:{},y:{},width:{},height:{},rotation:{}".format(np.round(x),
                                                                  np.round(y), np.round(width), np.round(height), np.round(rotation))
            print(s)

            # Kutucuk oluşturma işlemi
            box = cv2.boxPoints(rect)
            box = np.int64(box)

            # Görüntünün merkezini bulmak için moment kullanıyoruz.
            M = cv2.moments(c)

            # Momentin ilgili değerleri ile işlem yapılarak merkez bulunur.
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

            # Konturu çizdirme işlemi (Sarı renk)
            cv2.drawContours(img_original, [box], 0, (0, 255, 255), 2)

            # Merkeze nokta çizdirme işlemi (-1 içini doldurma anlamı taşır.)
            # Pembe renkli
            cv2.circle(img=img_original, center=center, radius=5,
                       color=(255, 0, 255), thickness=-1)

            # Bilgileri ekrana yazdırma işlemi
            cv2.putText(img_original, s, (20, 50),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

            # Takip algoritması (basit)(Yeşil Renkli)
            pts.appendleft(center)

            for i in range(1, len(pts)):

                if pts[i-1] is None or pts[i] is None:
                    continue
                cv2.line(img_original, pts[i-1], pts[i], (0, 255, 0), 3)

            # Yaptıklarımızı inceleyelim.
            cv2.imshow("Orijinal Tespit", img_original)
            sleep(0.01)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
