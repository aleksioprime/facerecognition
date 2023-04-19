from fer import FER
import cv2

detector = FER()
video_capture = cv2.VideoCapture('test_faces_video_1.mp4')
count = 0
number = 0
while True:
    success, frame = video_capture.read()
    if not success:
        print("Ошибка подключения к камере!!!")
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_emotions(rgb_frame)
    for face in faces:
        if number % 30 == 0:
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 150), thickness=2)
            crop_face = frame[y:y + h, x:x + w]
            cv2.imwrite(f'faces/face-{count}.jpg', crop_face)
            count += 1
        number += 1
    # cv2.imshow('Video', frame)
    if cv2.waitKey(1) == ord(' '):
        break
video_capture.release()
cv2.destroyAllWindows()