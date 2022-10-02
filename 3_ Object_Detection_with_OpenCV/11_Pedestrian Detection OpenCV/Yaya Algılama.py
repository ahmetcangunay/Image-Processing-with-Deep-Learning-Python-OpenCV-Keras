import cv2
import os

images = [i for i in os.listdir() if i.endswith(".jpg")]

# Hog tanımlayıcısı
hog = cv2.HOGDescriptor()

# Tanımlayıcıya SVM (Support Vector Machine) ekleme işlemi
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for image in images:

    img = cv2.imread(image)

    (rects, weights) = hog.detectMultiScale(img, padding=(8, 8), scale=1.05)

    for (idx, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, "Yaya {}".format(idx + 1),
                    (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.55, (255, 0, 0), 2)

    cv2.imshow("Yaya: ", img)

    if cv2.waitKey(0) & 0xFF == ord('q'):

        cv2.destroyAllWindows()
        continue
