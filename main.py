import requests
import base64


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
