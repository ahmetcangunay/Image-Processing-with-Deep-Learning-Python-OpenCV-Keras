# -*- coding: utf-8 -*-
"""
1) Veri seti oluþturma
    -Pozitif resimler(Tespit etmek istediğimiz obje)
    -Negatif resimler(Herhangi başka bir şey)
2) Cascade oluşturabilmek için programı indirme
3) Cascade oluşturma
4) Cascade'i kullanarak tespit algoritması yazma
"""

import cv2
import os

# Resimlerin depolanacaÄÄ± klasÃ¶r
path = "images"

# Resimlerin boyutu
imgWidth = 180
imgHeight = 120

# Video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
cap.set(10, 180)  # Brightness

# KlasÃ¶rler oluÅturma iÅlemi
global countFolder


def saveDataFunc():
    global countFolder
    countFolder = 0

    while os.path.exists(path + str(countFolder)):
        countFolder += 1

    os.makedirs(path + str(countFolder))


saveDataFunc()

count = 0
countSave = 0

while True:

    ret, frame = cap.read()

    if ret:

        # Resimlerin boyutunu sonradan ayarlÄ±yoruz ki kamera ayarlarÄ±na
        # dokunmayalÄ±m.
        frame = cv2.resize(frame, (imgWidth, imgHeight))

        # Her resmi depolamaya gerek yok 5 resimde bir yeterli.
        if count % 5 == 0:
            cv2.imwrite(path+str(countFolder)+"/" +
                        str(countSave)+"_"+".png", frame)
            countSave += 1
            print(countSave)
        count += 1

        cv2.imshow("Image {}".format(countSave), frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or countSave == 50:
        break


cap.release()
cv2.destroyAllWindows()
