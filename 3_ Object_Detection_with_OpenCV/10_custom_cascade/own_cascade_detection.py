import cv2

object_name = "Kalem Ucu"
frame_width = 280
frame_height = 360
color = (255, 0, 0)

cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

# Trackbar oluşturma işlemi (Scale ve Neighbors değerleri ayarlayabilmek için.)
cv2.namedWindow("Sonuc")
cv2.resizeWindow("Sonuc", width=frame_width, height=frame_height + 100)

# Boş bir fonksiyon(empty) dönebilmek için fonksiyon oluşturuyoruz.


def empty(a):
    pass


# Scale'i değiştirme işlemi
cv2.createTrackbar("Scale", "Sonuc", 400, 1000, empty)
cv2.createTrackbar("Neighbor", "Sonuc", 4, 50, empty)

# Cascade classifier
cascade = cv2.CascadeClassifier("cascade.xml")

while True:

    # Read frame (or image)
    ret, frame = cap.read()

    if ret:

        # Convert BGR to GRAY
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detection parameters
        scale_value = 1 + \
            (cv2.getTrackbarPos(trackbarname="Scale", winname="Sonuc")/1000)

        neighbor_value = cv2.getTrackbarPos(
            trackbarname="Neighbor", winname="Sonuc")

        # Detection
        rects = cascade.detectMultiScale(
            gray_frame, scale_value, neighbor_value)

        for (x, y, w, h) in rects:

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, object_name, (x, y-10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)

        cv2.imshow("Sonuc", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
