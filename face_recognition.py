from fer import FER
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

video_capture = cv2.VideoCapture('test_video.mp4')
model = load_model('tm_models/model_signs.h5')
labels = ['interested', 'uninterested']
cap_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
_, model_width, model_height, _ = model.get_config()["layers"][0]["config"]["batch_input_shape"]

df = pd.DataFrame(columns=['frame', 'peoples', 'interested', 'percent'])
time = 0
detector = FER()
print("Форма входного слоя модели: {}x{}".format(model_width, model_height))
emotions = {}
timeline = []
# while True:
#     success, frame = video_capture.read()
#     if not success:
#         print("Ошибка подключения к камере!!!")
#         break
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     faces = detector.detect_emotions(rgb_frame)
#     interested = 0
#     for face in faces:
#         x, y, w, h = face['box']
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 150), thickness=2)
#         max_emotion = max(face['emotions'], key=face['emotions'].get)
#         cv2.putText(frame,
#                     text=max_emotion,
#                     org=(x, y - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                     fontScale=0.6, color=(255, 255, 255),
#                     thickness=1, lineType=cv2.LINE_AA)
#         if max_emotion in ['neutral', 'happy']:
#             interested += 1
#     percent = round(interested / len(faces) * 100)
#     df = df.append({'frame': time, 'peoples': len(faces), 'interested': interested, 'percent': percent},
#                    ignore_index=True)
#     time += 1
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(1) == ord(' '):
#         break


df.to_excel("output.xlsx")
print(f"Заинтересованность группы составляет {round(df.percent.mean())}%")

plt.figure(figsize=(16, 10), dpi=80)
plt.plot(df.frame, df.percent, color='tab:blue', label='Interested faces')
plt.ylim(0, 100)
plt.title("График заинтересованности аудитории", fontsize=22)
plt.savefig('test.png')
plt.show()

video_capture.release()
cv2.destroyAllWindows()