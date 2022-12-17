import requests
import base64


def get_iamge_code():
    response = requests.get('https://check.gibdd.ru/captcha').json()
    print(response)
    base_code = response['base64jpg']
    return base_code


def jpg_from_base64(name: str, code):
    png_recover = base64.b64decode(code)
    with open(f'{name}.jpg', 'wb') as f:
        f.write(png_recover)
        f.close()
    print(f'File {name}.jpg saved')


if __name__ == "__main__":
    code = get_iamge_code()
    jpg_from_base64('temp4', code)
