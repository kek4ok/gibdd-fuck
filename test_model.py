from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense

import multiprocessing
import os
from random import shuffle
from typing import Tuple, Optional, List

import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Functional
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt


def _provide_data(for_train_percent, num_of_threads: Optional[int] = 2):
    paths_to_images = []

    for el in range(0, 10):
        paths_to_images.extend(f'bot_tg/imgs/training/{el}/{image_name}' for image_name in
                               os.listdir(f'bot_tg/imgs/training/{el}'))

    shuffle(paths_to_images)

    size = 100
    train_size = int(size * for_train_percent / 100)
    test_size = size - train_size
    chunk_size = size // num_of_threads

    manager = multiprocessing.Manager()
    data = manager.list()
    processes = []
    for i in range(chunk_size):
        start = i * chunk_size
        stop = (i + 1) * chunk_size
        process = multiprocessing.Process(target=_provide_images, args=(paths_to_images[start:stop], data))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    all_images = [value[0] / 255 for value in data]
    all_labels = [value[1] for value in data]
    x_train, y_train = all_images[:train_size], all_labels[:train_size]
    x_test, y_test = all_images[-test_size:], all_labels[-test_size:]
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def _provide_images(paths_to_image: List[str], all_images: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    for path_to_image in paths_to_image:
        digit = int(path_to_image.split('/')[-2])
        digit_vectorize = np.zeros(shape=(10,))
        digit_vectorize[digit] = 1

        img = cv2.imread(path_to_image)
        all_images.append((img, digit_vectorize))


if __name__ == '__main__':
    multiprocessing.freeze_support()

    model = keras.models.load_model("hyi_1_test.h5")

    x_train, y_train, x_test, y_test = _provide_data(for_train_percent=100)
    while 1:
        n = int(input())
        x = np.expand_dims(x_train[n], axis=0)
        res = model.predict(x)
        print(res)
        print(f"Что увидел робот:{np.argmax(res)}")
        print(f"Что на самом деле:{y_train[n]}")
        plt.imshow(x_train[n], cmap=plt.cm.binary)
        plt.show()
