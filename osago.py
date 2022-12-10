import datetime

import requests

url = 'https://b2c.pampadu.ru/b2c/info/getAutoCodeInfo'


def get_vin(car_number: str):
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3',
        'Connection': 'keep-alive',
        'Content-Length': '0',
        'Content-Type': 'application/json',
        'Cookie': 'amp_1ff27c=R0GHttzQ8UJ-V8-mzNN7Kh.MDk5YmMzNDQtM2ZkMy00MWFhLWIyODctMTIyYTczYjI2ZDJm..1gju630p3.1gju6berp.4.2.6',
        'Host': 'b2c.pampadu.ru',
        'Origin': 'https://b2c.pampadu.ru',
        'Referer': 'https://b2c.pampadu.ru/index.html',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:107.0) Gecko/20100101 Firefox/107.0',
        'UserId': '099bc344-3fd3-41aa-b287-122a73b26d2f',
        'widgetid': '49a973bd-2d7c-4b9b-9c28-d986d7757983'
    }

    params = {
        "licensePlate": f"{car_number}",
        "token": {"isTrusted": True}
    }

    response = requests.post(url, headers=headers, params=params).json()
    print(response)

    return response['data']['report']['vin']
