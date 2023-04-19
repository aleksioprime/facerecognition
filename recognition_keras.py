import cv2
from tensorflow.keras.models import load_model
import numpy as np
import time
model = load_model('tm_models/model_signs.h5')
labels = open("tm_models/labels.csv").read().strip().split("\n")
labels = [l.split(",")[1] for l in labels]
cap = cv2.VideoCapture(0)
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
_, model_width, model_height, _ = model.get_config()["layers"][0]["config"]["batch_input_shape"]
print("Форма входного слоя модели: {}x{}".format(model_width, model_height))
while True:
    success, frame = cap.read()
    if success:
        frame = cv2.flip(frame, 1)
        frame_cut = frame[:, (cap_width - cap_height) // 2:(cap_width - cap_height) // 2 + cap_height]
        image = cv2.resize(frame_cut, (model_width, model_height))
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        start_time = time.time()
        prediction = model.predict(image)
        ms = (time.time() - start_time) * 1000
        name = labels[prediction.argmax(axis=1)[0]]
        cv2.rectangle(frame, ((cap_width - cap_height) // 2, 0),
                      ((cap_width - cap_height) // 2 + cap_height, cap_height), (255, 0, 0), 3)
        cv2.putText(frame, "{}".format(name), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "{}".format(round(ms, 1)), (10, cap_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.imshow("Result image", frame)
        if cv2.waitKey(1) == ord(' '):
            break
    else:
        print("Error!")
        break
cv2.destroyAllWindows()
cap.release()