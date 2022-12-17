import cv2
import numpy as np
import os.path
import glob


CAPTCHA_IMAGE_FOLDER = "cap"
OUTPUT_FOLDER = "readi"


captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}
counts_name = 0

for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0].split('_')[0]

    image = cv2.imread(captcha_image_file)
    cv2.imshow("cropped", image)
    cv2.waitKey(0)

def huy_clean(image_mask):
#даелает маску
    image_mask = image.copy()
    for column in range(1, image_mask.shape[1] - 1):
        if 2 < column < 148:
            for row in range(1, image_mask.shape[0] - 1):
                if 2 < row < 78:
                    if (image_mask[row, column][0] <= 90 and image_mask[row, column][1] <= 90 and image_mask[row, column][2] <= 90):
                        image_mask[row, column][0] = 255
                        image_mask[row, column][1] = 255
                        image_mask[row, column][2] = 255
                    else:
                        image_mask[row, column][0] = 0
                        image_mask[row, column][1] = 0
                        image_mask[row, column][2] = 0

    cv2.imshow("cropped", image_mask)
    cv2.waitKey(0)
# Делаем маску больше
    #ret, image_mask_BN = cv2.threshold(image_mask, 254, 255, cv2.THRESH_BINARY)
    image_mask_BN = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), 'uint8')
    mask = cv2.dilate(image_mask_BN, kernel, iterations = 1)
    ready_img = cv2.inpaint(image, mask, 10, cv2.INPAINT_TELEA)

#Черно белое
    image_grey = cv2.cvtColor(ready_img, cv2.COLOR_BGR2GRAY)
    ret, image_TH = cv2.threshold(image_grey, 157, 255, cv2.THRESH_BINARY)

    cv2.imshow("cropped", image_TH)
    cv2.waitKey(0)
    for i in range(2):
        for column in range(1, image_TH.shape[1] - 1):
            for row in range(1, image_TH.shape[0] - 1):
                if image_TH[row, column] == 0:
                    if (image_TH[row+1, column] == 255 and image_TH[row-1, column] == 255):
                        image_TH[row, column] = 255
                    if (image_TH[row, column+1] == 255 and image_TH[row, column-1] == 255):
                        image_TH[row, column] = 255
                if (5 < column < 146 and 5 < row < 74 ):
                    if (image_TH[row+2, column] == 255 and image_TH[row-1, column] == 255):
                        image_TH[row, column] = 255
                    if (image_TH[row, column+2] == 255 and image_TH[row, column-1] == 255):
                        image_TH[row, column] = 255


    for column in range(1, image_TH.shape[1] - 1):
        for row in range(1, image_TH.shape[0] - 1):
            if (0 <column< 5):
                image_TH[row, column] = 255
            if ( 145< column < 160):
                image_TH[row, column] = 255
            if (0 <row< 5):
                image_TH[row, column] = 255
            if (74 < row < 80):
                image_TH[row, column] = 255

    return image_TH
def huy_cut(image_TH):
    # нарезка
    x = 0
    y = 0
    x_max = 0
    y_max = 0
    for column in range(1, image_TH.shape[1] - 1):
        for row in range(1, image_TH.shape[0] - 1):
            if (image_TH[row, column] == 0):
                x = column
                break
        if (x != 0):
            break
    for row in range(1, image_TH.shape[0] - 1):
        for column in range(1, image_TH.shape[1] - 1):
            if (image_TH[row, column] == 0):
                y = row
                break
        if (y != 0):
            break
    for column in range(image_TH.shape[1] - 1, -1, -1):
        for row in range(image_TH.shape[0] - 1, -1, -1):
            if (image_TH[row, column] == 0):
                x_max = column
                break
        if (x_max != 0):
            break
    for row in range(image_TH.shape[0] - 1, -1, -1):
        for column in range(image_TH.shape[1] - 1, -1, -1):
            if (image_TH[row, column] == 0):
                y_max = row
                break
        if (y_max != 0):
            break
    crop_img_TH = image_TH[y:y_max, x:x_max]


    (x_short, y_short, w, h) = cv2.boundingRect(crop_img_TH)
    one_img_w = round(w / 5)
    array = []
    for el in range(5):
        img_belia = np.full((60, 30), 255, np.uint8)
        one_img = crop_img_TH[0:h, el * one_img_w:(el + 1) * one_img_w]
        for row in range(1, one_img.shape[0] - 1):
            for col in range(1, one_img.shape[1] - 1):
                img_belia[row, col] = one_img[row, col]
        array.append(img_belia)
    return array