import cv2
import time

# Video değişkeni oluşturma

video_name = "MOT17-04-DPM.mp4"

# Video içeri aktarma: capture, cap

cap = cv2.VideoCapture(video_name)

# Videonun genişliği ve yüksekliğini gösterme

print("Genişlik:", cap.get(3))
print("Yükseklik:", cap.get(4))

# Videonun içeri aktarılıp aktarılmadığını denetlemek

if not cap.isOpened():
    print("Video Yüklenirken Hata Meydana Geldi!")
else:
    print("Video Başarıyla Yüklendi.")

# Videoyu "Okuma" işlemi (Frame: Videonun içindeki herbir resim Return: İşlemin başarılı olup olmadığını döner.)
# False or True

while True:
    ret, frame = cap.read()

    if ret:
        time.sleep(0.01)  # UYARI: Kullanmazsak video çok hızlı akar!
        cv2.imshow("Video", frame)  # Herbir Frame teker teker gösterilir.
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Stop Capture
cv2.destroyAllWindows()
