import cv2
import numpy as np
import os.path
import glob

CAPTCHA_IMAGE_FOLDER = "../bot_tg/imgs"
OUTPUT_FOLDER = "readi"


def remove_lines(image):
    #image = cv2.imread(img_path)

    # даелает маску
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
    # ret, image_mask_BN = cv2.threshold(image_mask, 254, 255, cv2.THRESH_BINARY)
    image_mask_BN = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), 'uint8')
    mask = cv2.dilate(image_mask_BN, kernel, iterations=1)
    ready_img = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)

    # Черно белое
    image_grey = cv2.cvtColor(ready_img, cv2.COLOR_BGR2GRAY)
    ret, image_TH = cv2.threshold(image_grey, 157, 255, cv2.THRESH_BINARY)

    return image_TH


# сохранени
'''
save_path = os.path.join(OUTPUT_FOLDER)
name = captcha_correct_text + ".jpg"
all_path = os.path.join(save_path, name)
cv2.imwrite(all_path, image_TH)
'''

'''#нарезка
    x = 0
    y = 0
    x_max = 0
    y_max = 0
    for column in range(1,image_TH.shape[1] - 1):
        for row in range(1, image_TH.shape[0] - 1):
            if (image_TH[row, column] == 0):
                x = column
                break
        if (x != 0):
            break
    for row in range(1,image_TH.shape[0] - 1):
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
    cv2.imshow("cropped", crop_img_TH)
    cv2.waitKey(0)
    (x_short, y_short, w, h) = cv2.boundingRect(crop_img_TH)
    print(x_short)
    print(y_short)
    print(w)
    print(h)
    one_img_w = round(w/5)
    sdvig = 0
    minsdvig = 0
    lastMove = 0
    for el in range(5):
        one_img =crop_img_TH[0:h, el*one_img_w:(el+1)*one_img_w]
        y_hith = 0
        for row in range(1, one_img.shape[0] - 1):
            for col in range(10, one_img.shape[1] - 9):
                if (one_img[row, col] == 0):
                    y_hith = row
                    break
            if (y_hith != 0):
                break


        if (w - y_hith)/one_img_w < 4.5:
            one_img = crop_img_TH[0:h, el * one_img_w:(el + 1) * round(one_img_w*0.75)]
            sdvig = sdvig + round(one_img_w*0.75)
            if el != 0:
                minsdvig = minsdvig + lastMove
            lastMove = round(one_img_w*0.75)
        else:
            sdvig = sdvig + one_img_w
            if el != 0:
                minsdvig = minsdvig + lastMove
            lastMove = one_img_w



        one_img = crop_img_TH[0:h, minsdvig:sdvig]
        cv2.imshow("cropped", one_img)
        cv2.waitKey(0)
    #pixel_matrix_RGB = image_orig.copy()
'''

'''
    for column in range(1, image.shape[1] - 1):
        for row in range(1, image.shape[0] - 1):
            if (image[row, column][0] <= 60 and image[row, column][1] <= 60 and image[row, column][2] <= 60):
                image[row, column][0] = 255
                image[row, column][1] = 255
                image[row, column][2] = 255
            if (image[row, column][0] >= 140 and image[row, column][1] >= 200 and image[row, column][2] >= 200):
                image[row, column][0] = 255
                image[row, column][1] = 255
                image[row, column][2] = 255

    cv2.imshow('None approximation', image)
    cv2.waitKey(0)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 157, 255, cv2.THRESH_BINARY)

    pixel_matrix_WB = image.copy()
    for column in range(1, image.shape[1] - 1):
        for row in range(1, image.shape[0] - 1):
            if pixel_matrix_WB[row, column] == 0 and pixel_matrix_WB[row, column - 1] == 255 and pixel_matrix_WB[row, column + 1] == 255:
                pixel_matrix_WB[row, column] = 255
            if pixel_matrix_WB[row, column] == 0 and pixel_matrix_WB[row - 1, column] == 255 and pixel_matrix_WB[row + 1, column] == 255:
                pixel_matrix_WB[row, column] = 255

    save_path = os.path.join(OUTPUT_FOLDER)
    name = captcha_correct_text + ".jpg"
    all_path = os.path.join(save_path, name)

    cv2.imshow('None approximation', pixel_matrix_WB)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(image=pixel_matrix_WB, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    image_copy = pixel_matrix_WB.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 255), thickness=1,
                     lineType=cv2.CHAIN_APPROX_NONE)

    cv2.imshow('None approximation', image_copy)
    cv2.waitKey(0)
    cv2.imwrite(all_path, pixel_matrix_WB)
'''
