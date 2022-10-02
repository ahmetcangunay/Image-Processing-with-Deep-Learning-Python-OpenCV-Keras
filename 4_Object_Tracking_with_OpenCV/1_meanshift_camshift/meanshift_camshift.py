import cv2

# Capturing
cap = cv2.VideoCapture(0)

# Bir adet frame okuma
ret, frame = cap.read()

# Eğer kamera okuma işlemi gerçekleştiremezse hata versin.
if not ret:
    print("Uyarı!")

# Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_rects = face_cascade.detectMultiScale(frame)

(face_x, face_y, w, h) = tuple(face_rects[0])

# meanshift algoritması girdisi(argümanı)
track_window = (face_x, face_y, w, h)

# Region of Interest (Yüzü tespit ettiğimiz alanlar) roi = face
roi = frame[face_y:face_y + h, face_x:face_x + w]

# HSV color changing
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Histogram hesaplama (Takip için histogram gerekli.)
roi_hist = cv2.calcHist([hsv_roi], channels=[0], mask=None,
                        histSize=[180], ranges=[0, 180])

# Histogramı normalize edelim.
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Takip için gerekli durdurma kriterleri
# @param count = hesaplanacak maksimum öge saysı
# @param eps = değişiklik katsayısı

# 5 yineleme veya bir tane epsilon
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)

# Takibi başlatma
while True:

    ret, frame = cap.read()

    if ret:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Back Project hesaplama (Historamı bir görüntüde bulmak için
        # kullanıyoruz.)
        # Piksel karşılaştırma işlemi
        dst = cv2.calcBackProject(images=[hsv], channels=[
                                  0], hist=roi_hist, ranges=[0, 180], scale=1)

        ret, track_window = cv2.meanShift(
            probImage=dst, window=track_window, criteria=term_crit)

        # Nesnenin yeni konumu -> track_window
        x, y, w, h = track_window

        # Yeni resim üzerinde istenilenleri çizdirme
        img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow("Tracking", img2)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
