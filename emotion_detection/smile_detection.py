#smile detection using opencv
# Importing the libraries
import cv2
# Loading the cascades
#for detecting face
facecas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#for detecting eyes
eyecas = cv2.CascadeClassifier('haarcascade_eye.xml')
#for detecting smile on face
smilecas = cv2.CascadeClassifier('haarcascade_smile.xml')

# Defining a function that will do the detections
def detect_face(gray, frame):
    faces = facecas.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyecas.detectMultiScale(roi_gray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)
        smiles = smilecas.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "press q to exit screen" ,(100,450), font, 1,(255,255,255),1,cv2.LINE_AA)
    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smile = detect_face(gray, frame)
    cv2.imshow('face eye and smile detection ',smile)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
