import numpy as np
import os
import glob
import cv2

NUM_OF_IMAGES = 5
CAPTCHA_IMAGE_FOLDER = "imgs/readi"
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))

counter = 0
def cut_numbers(img: np.ndarray, correct_name):
    global counter
    w = img.shape[1] // NUM_OF_IMAGES  # Ширина одной картинки
    numbers = []
    for i in range(NUM_OF_IMAGES):
        numbers.append(img[:, i * w: i * w + w])

    for cnt, letter in enumerate(correct_name):
        save_path = f'bot_tg/imgs/training/{letter}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        p = os.path.join(save_path, f"{counter}.jpg")
        cv2.imwrite(p, numbers[cnt])
        counter += 1

for img in captcha_image_files:
    filename = os.path.basename(img)
    captcha_correct_text = os.path.splitext(filename)[0]
    print(img)
    print(captcha_correct_text)
    image = cv2.imread(img)
    cut_numbers(image, captcha_correct_text)
