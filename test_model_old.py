import datetime

from captcha import *
import cv2
import numpy as np
from img_2 import huy_cut, huy_clean
from tensorflow import keras
import matplotlib.pyplot as plt


def thresh_image(image):
    # image = cv2.imread(img_path)

    image_mask = image.copy()
    for column in range(1, image_mask.shape[1] - 1):
        for row in range(1, image_mask.shape[0] - 1):
            if (image_mask[row, column][0] <= 90 and image_mask[row, column][1] <= 90 and image_mask[row, column][
                2] <= 90):
                image_mask[row, column][0] = 255
                image_mask[row, column][1] = 255
                image_mask[row, column][2] = 255
            else:
                image_mask[row, column][0] = 0
                image_mask[row, column][1] = 0
                image_mask[row, column][2] = 0

    # Делаем маску больше
    image_mask_BN = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), 'uint8')
    mask = cv2.dilate(image_mask_BN, kernel, iterations=1)
    ready_img = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)

    # Черно белое
    image_grey = cv2.cvtColor(ready_img, cv2.COLOR_BGR2GRAY)
    ret, image_TH = cv2.threshold(image_grey, 157, 255, cv2.THRESH_BINARY)

    return image_TH


def cut_image(img: np.ndarray):
    w = img.shape[1] // 5  # Ширина одной картинки
    numbers = []
    for i in range(5):
        numbers.append(img[:, i * w: i * w + w])

    return numbers


def get_captcha():
    response = requests.get('https://check.gibdd.ru/captcha').json()
    base_code = response['base64jpg']
    png_recover = base64.b64decode(base_code)
    np_data = np.frombuffer(png_recover, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    token = response['token']

    return img, token


def get_history(answer: str, token: str):
    url = 'https://xn--b1afk4ade.xn--90adear.xn--p1ai/proxy/check/auto/history'
    payload = {
        "vin": "XTA210530W1730856",
        "checkType": "history",
        "captchaWord": answer,
        "captchaToken": token
    }
    response = requests.post(url, data=payload).json()

    return response


def recognize_captcha(numbers: list):
    answer = ''
    for i in numbers:
        x = cv2.cvtColor(i, cv2.COLOR_GRAY2RGB)
        x = x / 255
        x = np.expand_dims(x, axis=0)
        res = model.predict(x, verbose=0)
        print(np.argmax(res))
        answer += str(np.argmax(res))

    return answer

def try_to_solve():
    counter = 0
    while 1:
        counter += 1
        img_path, token = get_captcha()
        #plt.imshow(img_path)
        #plt.show()
        img = huy_clean(img_path)
        numbers = huy_cut(img)
        answer = recognize_captcha(numbers)
        response = get_history(answer, token)
        if response.get('code') == None:
            #print(response)
            print(f'Captcha was solved in {counter} attempts')
            break

if __name__ == '__main__':
    model = keras.models.load_model('model_99.h5')
    for i in range(2):
        time_start = datetime.datetime.now()
        try_to_solve()
        print(f"Time requiers: {datetime.datetime.now() - time_start}")
