from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import layers, models
from tensorflow import lite
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import cv2
import datetime


# КОНФИГУРАЦИЯ НЕЙРОННОЙ СЕТИ
def get_model(input_size, classes=7):
    model = models.Sequential()   

    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape =input_size))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(classes, activation='softmax'))

    # компиляция модели
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ЗАГРУЗКА И ПРЕОБРАЗОВАНИЕ ИЗОБРАЖЕНИЙ
def dataset_load(im_paths, width, height, verbose):
    data = []
    labels = []
    for (i, im_path) in enumerate(im_paths):
        # загружаем изображение в переменную image
        image = cv2.imread(im_path)
        # определяем класс изображения из строки пути
        # формат пути: ../dataset/{class}/{image}.jpg
        label = im_path.split(os.path.sep)[-2]
        # изменяем размер изображения на заданный (изображение должно быть квадратным)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        # переводим изображение в массив numpy
        image_array = img_to_array(image, data_format=None)
        # добавляем массив изображения в список data
        data.append(image_array)
        # добавляем в список labels метку соответствующего изображения из списка data
        labels.append(label)
        # выводим на экран количество обработанных изображений в периодичностью verbose
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] Обработано {}/{}".format(i + 1, len(im_paths)))
    # возвращаем numpy массивы data и labels
    return (np.array(data), np.array(labels))


# 1. ПОДГОТОВКА ДАННЫХ
# указываем название каталога набора данных в папке datasets
dataset_name = "faces"
# определяем пути набора данных, сохранения графика обучения и модели нейронной сети keras
dataset_path = os.path.join("datasets", dataset_name)
name_labels = ['interested', 'uninterested']
num_classes = len(name_labels)
plot_name = "{}_output/{}_plot.png".format(dataset_name, dataset_name)
weights_name = "{}_output/{}_weights.h5".format(dataset_name, dataset_name)
tflite_name = "{}_output/{}_weights.tflite".format(dataset_name, dataset_name)
# загружаем набор данных с диска, преобразуя изображения в массив
# и масштабируя значения пикселей из диапазона [0, 255] в диапазон [0, 1]
start_time = time.time()
image_paths = list(paths.list_images(dataset_path))
print("[INFO] Загрузка изображений ...")
(data, labels) = dataset_load(image_paths, width=48, height=48, verbose=500)
data = data.astype("float") / 255.0
# разделяем данные на обучающий и тестовый наборы (75% и 25%)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
print("[INFO] Форма матрицы признаков: {}".format(data.shape))
print("[INFO] Размер матрицы признаков: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))
# преобразуем метки из целых чисел в векторы
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)
print("[INFO] Время подготовки данных: {} сек".format(round(time.time() - start_time, 2)))


# 2.СБОРКА И КОМПИЛЯЦИЯ МОДЕЛИ НЕЙРОННОЙ СЕТИ
print("[INFO] Компиляция модели...")
model = get_model((48,48,1), 2)

# 3.ФОРМИРОВАНИЕ ДОПОЛНИТЕЛЬНЫХ ПАРАМЕТРОВ ОБУЧЕНИЯ
# Определение коллбэков для обучения нейронной сети
log_dir = "checkpoint/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint = ModelCheckpoint(filepath=weights_name,
                             save_best_only=True,
                             verbose=1,
                             mode='min',
                             moniter='val_loss')       
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.2, 
                              patience=6, 
                              verbose=1, 
                              min_delta=0.0001)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
csv_logger = CSVLogger('training.log')
callbacks = [checkpoint, reduce_lr, csv_logger]
# настройка метода увеличения выборки данных для обучения через модификацию существующих данных (аугментация)
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# 4. ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ
num_epochs = 30
print("[INFO] Обучение нейронной сети...")
start_time = time.time()
hist = model.fit(aug.flow(trainX, trainY, batch_size=32),
                 validation_data=(testX, testY),
                 batch_size=64,
                 epochs=num_epochs,
                 callbacks=callbacks, 
                 verbose=0)
print("[INFO] Время обучения: {} сек".format(round(time.time() - start_time, 2)))

# 5.ОЦЕНКА МОДЕЛИ НЕЙРОННОЙ СЕТИ
print("[INFO] Оценка нейронной сети...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=name_labels))
# построение и сохранение графика потерь и точности тренировок
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, num_epochs), hist.history["loss"], label="train_loss")
plt.plot(np.arange(0, num_epochs), hist.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, num_epochs), hist.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, num_epochs), hist.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(plot_name)

# 6. СОХРАНЕНИЕ МОДЕЛИ НЕЙРОННОЙ СЕТИ
print("[INFO] Сохранение модели TFLite с квантованием...")
# конвертирование модели keras в квантованную модель tflite
converter = lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [lite.Optimize.DEFAULT]
tflite_model = converter.convert()
# сохранение модели tflite.
with open(tflite_name, 'wb') as f:
    f.write(tflite_model)
