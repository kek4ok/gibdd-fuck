import numpy as np
import requests
import cv2
import base64


def get_numbers(img: np.ndarray):
    w = img.shape[1] // 5  # Ширина одной картинки
    numbers = []
    for i in range(5):
        numbers.append(img[:, i * w: i * w + w])

    return numbers


def get_captcha():
    response = requests.get('https://check.gibdd.ru/captcha').json()
    print(response)
    base_code = response['base64jpg']
    png_recover = base64.b64decode(base_code)
    np_data = np.frombuffer(png_recover, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    token = response['token']

    return img, token


def get_history(answer: str, token: str, vin: str):
    url = 'https://xn--b1afk4ade.xn--90adear.xn--p1ai/proxy/check/auto/history'
    payload = {
        "vin": f"{vin}",
        "checkType": "history",
        "captchaWord": answer,
        "captchaToken": token
    }
    response = requests.post(url, data=payload).json()

    return response

def get_diagnostic(answer: str, token: str, vin: str):
    url = 'https://xn--b1afk4ade.xn--90adear.xn--p1ai/proxy/check/auto/diagnostic'
    payload = {
        "vin": f"{vin}",
        "checkType": "diagnostic",
        "captchaWord": answer,
        "captchaToken": token
    }
    response = requests.post(url, data=payload).json()

    return response

def get_restrict(answer: str, token: str, vin: str):
    url = 'https://xn--b1afk4ade.xn--90adear.xn--p1ai/proxy/check/auto/restrict'
    payload = {
        "vin": f"{vin}",
        "checkType": "restrict",
        "captchaWord": answer,
        "captchaToken": token
    }
    response = requests.post(url, data=payload).json()

    return response

def get_dtp(answer: str, token: str, vin: str):
    url = 'https://xn--b1afk4ade.xn--90adear.xn--p1ai/proxy/check/auto/dtp'
    payload = {
        "vin": f"{vin}",
        "checkType": "dtp",
        "captchaWord": answer,
        "captchaToken": token
    }
    response = requests.post(url, data=payload).json()

    return response

def get_wanted(answer: str, token: str, vin: str):
    url = 'https://xn--b1afk4ade.xn--90adear.xn--p1ai/proxy/check/auto/wanted'
    payload = {
        "vin": f"{vin}",
        "checkType": "wanted",
        "captchaWord": answer,
        "captchaToken": token
    }
    response = requests.post(url, data=payload).json()

    return response


