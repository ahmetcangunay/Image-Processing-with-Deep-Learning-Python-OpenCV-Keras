import cv2
import os

img_list = [i for i in os.listdir() if i.endswith(".jpg")]
print(img_list)

for j in img_list:

    img = cv2.imread(j)
    cv2.imshow(j, img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

    # scaleFactor, resme ne kadar zoom yapıcağımızı belirtir.
    rects = detector.detectMultiScale(
        gray_img, scaleFactor=1.045, minNeighbors=2)

    for (idx, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(img, f"Kedi {idx+1}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow("Detection", img)

    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
