import datetime

from captcha import *
import cv2
import numpy as np
from clean_img.main import thresh_image, cut_image
from tensorflow import keras
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
        #plt.imshow(img_path)
        #plt.show()
        img = thresh_image(img_path)
        numbers = cut_image(img)
        answer = recognize_captcha(numbers)
        response = get_history(answer, token)
        if response.get('code') == None:
            #print(response)
            print(f'Captcha was solved in {counter} attempts')
            break


if __name__ == '__main__':
    model = keras.models.load_model('model_97_1.4.h5')
    for i in range(20):
        time_start = datetime.datetime.now()
        try_to_solve()
        print(f"Time requiers: {datetime.datetime.now() - time_start}")
