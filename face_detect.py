import cv2

face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_data.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 0, 0), 2)

    cv2.imshow('image', frame)

    k = cv2.waitKey(1)

    if k == ord('q'): #unicode
        break

camera.release()
cv2.destroyAllWindows()
