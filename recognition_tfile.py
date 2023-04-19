import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import time

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

# функция возвращает отсортированный массив результатов классификации
def classify_image(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

# interpreter = tflite.Interpreter(model_path='signs_output/signs_weights.tflite')
interpreter = tflite.Interpreter(model_path='tm_models/lite_model_signs.tflite')
labels = open("signs_labels.txt").read().strip().split("\n")
labels = [l.split(",")[1] for l in labels]
interpreter.allocate_tensors()
_, model_height, model_width, _ = interpreter.get_input_details()[0]['shape']
print("Форма входного слоя модели: {}x{}".format(model_width, model_height))

cap = cv2.VideoCapture(0)
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    success, frame = cap.read()
    if success:
        frame = cv2.flip(frame, 1)
        frame_cut = frame[:, (cap_width - cap_height) // 2:(cap_width - cap_height) // 2 + cap_height]
        image = cv2.resize(frame_cut, (model_width, model_height))
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        start_time = time.time()
        results = classify_image(interpreter, image)
        ms = (time.time() - start_time) * 1000
        label_id, prob = results[0]
        cv2.rectangle(frame, ((cap_width - cap_height) // 2, 0), ((cap_width - cap_height) // 2 + cap_height, cap_height), (255, 0, 0), 3)
        cv2.putText(frame, "{}".format(labels[label_id]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "{}".format(round(ms, 1)), (10, cap_height-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Result image", frame)
        if cv2.waitKey(1) == ord(' '):
            break
    else:
        print("Error!")
        break
cv2.destroyAllWindows()
cap.release()
