import requests
import base64


def get_iamge_code():
    response = requests.get('https://check.gibdd.ru/captcha').json()
    print(response)
    base_code = response['base64jpg']
    return base_code


def jpg_from_base64(name: str):
    code = get_iamge_code()
    png_recover = base64.b64decode(code)
    with open(f'imgs/{name}.jpg', 'wb') as f:
        f.write(png_recover)
        f.close()
    print(f'File {name}.jpg saved')


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

res = get_history("43954", "SgsJE4YwBcC85kpRbfMbDgQx9buyJUAiKIvxs4E+FMw=")
print(res.get('code'))
