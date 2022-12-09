import datetime

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense

import multiprocessing
import os
from random import shuffle
from typing import Tuple, Optional, List
from captcha import *
import cv2
import numpy as np
from clean_img.main import remove_lines
from keras.callbacks import ModelCheckpoint
from keras.models import Functional
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt


def recognize_captcha(numbers: list):
    answer = ''
    for i in numbers:
        x = cv2.cvtColor(i, cv2.COLOR_GRAY2RGB)
        x = x / 255
        x = np.expand_dims(x, axis=0)
        res = model.predict(x, verbose=0)  # use verbose=1 for showing logs
        answer += str(np.argmax(res))

    return answer


def try_to_solve():
    counter = 0
    while 1:
        counter += 1
        img_path, token = get_captcha()
        # plt.imshow(img_path)
        # plt.show()
        img = remove_lines(img_path)
        numbers = get_numbers(img)
        answer = recognize_captcha(numbers)
        response = get_history(answer, token)
        if response.get('code') == None:
            print(response)
            print(f'Captcha was solved in {counter} attempts')
            break


if __name__ == '__main__':
    model = keras.models.load_model('model_99.h5')
    time_start = datetime.datetime.now()
    try_to_solve()
    print(f"Time requiers: {datetime.datetime.now() - time_start}")
