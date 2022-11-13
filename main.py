import requests
import base64

def get_iamge_code():
    response = requests.get('https://check.gibdd.ru/captcha').json()
    base_code = response['base64jpg']
    return base_code

def jpg_from_base64(name: str):
    code = get_iamge_code()
    png_recover = base64.b64decode(code)
    with open(f'imgs/{name}.jpg', 'wb') as f:
        f.write(png_recover)
        f.close()
    print(f'File {name}.jpg saved')

for i in range(10000):
    jpg_from_base64(i)
