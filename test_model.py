import datetime

from captcha import *
import cv2
import numpy as np
#from clean_img.main import thresh_image, cut_image
from img_2 import huy_cut, huy_clean
from tensorflow import keras
from osago import get_vin

def recognize_captcha(numbers: list):
    answer = ''

    for i in numbers:
        x = cv2.cvtColor(i, cv2.COLOR_GRAY2RGB)
        x = x / 255
        x = np.expand_dims(x, axis=0)
        res = model.predict(x, verbose=0)  # use verbose=1 for showing logs
        answer += str(np.argmax(res))

    print(answer)
    return answer


def try_to_solve(vin):
    counter = 0
    while 1:
        try:
            counter += 1
            img_path, token = get_captcha()
            #plt.imshow(img_path)
            #plt.show()
            img = huy_clean(img_path)
            numbers = huy_cut(img)
            answer = recognize_captcha(numbers)
            response = get_history(answer, token, vin)
            if response.get('code') == None:
                print(response)
                print(f'Captcha was solved in {counter} attempts')
                break
        except Exception:
            pass


if __name__ == '__main__':
    model = keras.models.load_model('model_97_1.4.h5')
    #vin = get_vin(str(input("Введите номер авто: ")))
    time_start = datetime.datetime.now()
    try_to_solve('JMBXTGF3WDZ001984')
    print(f"Time requiers: {datetime.datetime.now() - time_start}")
