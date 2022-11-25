import os
import sys
import random
from loguru import logger


def get_random_image():
    f = open('../sended_captcha.txt', 'r+')
    sended_captcha = f.readlines()
    print(sended_captcha)
    all_imgs = os.listdir('imgs')
    img = random.choice(all_imgs)
    if img + '\n' not in sended_captcha and len(img) <= 9:
        img_path = os.path.join('imgs', img)
        f.write(img + '\n')
    else:
        get_random_image()

    return img_path


def rename_captcha(answer: str, img_path: str):
    """
    :param answer: resolved captcha from message
    :param img_path: path to captcha
    """
    old_name = img_path.split('\\')[1].split('.')[0]
    new_name = f'{answer}_{old_name}.jpg'
    new_path = os.path.join('imgs', new_name)
    os.rename(img_path, new_path)
    logger.info(f'File: {img_path} was renamed to {new_path}')


if __name__ == "__main__":
    pass