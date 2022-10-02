import cv2
import time

# Video capture (Buradaki 0'ın anlamı PC'deki default kamerayı kullanma işlemidir.)
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width, height)

# Video Kaydetme (Dosya Adı, Çerçeve Sıkıştırma (fourcc), FPS, Video Kaydedici Boyutu)
writer = cv2.VideoWriter("Video_kaydi.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 20, (width, height))

while True:
    ret, frame = cap.read()
    time.sleep(0.01)
    cv2.imshow("Video", frame)

    # Saving İşlemi
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
