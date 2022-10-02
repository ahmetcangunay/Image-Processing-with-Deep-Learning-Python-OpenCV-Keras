# Yüz tanıma, dijital görüntülerdeki insan yüzlerini tanımlayan çeşitli
# uygulamalarda kullanılan yöntemdir.

import cv2
import matplotlib.pyplot as plt
from time import sleep

einstein = cv2.imread("einstein.jpg", 0)

plt.figure()
plt.imshow(einstein, cmap="gray")
plt.axis("off")
plt.show()

# Sınıflandırıcı
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Koordinatları elde etme
face_rect = face_cascade.detectMultiScale(einstein)

for (x, y, w, h) in face_rect:
    cv2.rectangle(einstein, (x, y), (x+w, y+h), (255, 255, 255), 10)

plt.figure()
plt.imshow(einstein, cmap="gray")
plt.axis("off")
plt.show()

# Barcelona (Multi Face Detection)
barcelona = cv2.imread("barcelona.jpg", 0)

plt.figure()
plt.imshow(barcelona, cmap="gray")
plt.axis("off")
plt.show()

# Koordinatları elde etme
face_rect2 = face_cascade.detectMultiScale(barcelona, minNeighbors=7)

for (x, y, w, h) in face_rect2:
    cv2.rectangle(barcelona, (x, y), (x+w, y+h), (255, 255, 255), 10)

plt.figure()
plt.imshow(barcelona, cmap="gray")
plt.axis("off")
plt.show()

# %% Video (Kamera) ile yüz tanıma

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if ret:
        face_rect2 = face_cascade.detectMultiScale(frame, minNeighbors=7)

        for (x, y, w, h) in face_rect2:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 10)
            cv2.putText(frame, "Insan Evladi", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        cv2.imshow("Face Detection", frame)
        sleep(0.04)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
