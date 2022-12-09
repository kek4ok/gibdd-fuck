
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

#даелает маску
    image_mask = image.copy()
    for column in range(1, image_mask.shape[1] - 1):
        for row in range(1, image_mask.shape[0] - 1):
            if (image_mask[row, column][0] <= 90 and image_mask[row, column][1] <= 90 and image_mask[row, column][2] <= 90):
                image_mask[row, column][0] = 255
                image_mask[row, column][1] = 255
                image_mask[row, column][2] = 255
            else:
                image_mask[row, column][0] = 0
                image_mask[row, column][1] = 0
                image_mask[row, column][2] = 0

# Делаем маску больше
    #ret, image_mask_BN = cv2.threshold(image_mask, 254, 255, cv2.THRESH_BINARY)
    image_mask_BN = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), 'uint8')
    mask = cv2.dilate(image_mask_BN, kernel, iterations = 1)
    ready_img = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)

#Черно белое
    image_grey = cv2.cvtColor(ready_img, cv2.COLOR_BGR2GRAY)
    ret, image_TH = cv2.threshold(image_grey, 157, 255, cv2.THRESH_BINARY)

    #cv2.imshow("cropped", image_TH)
    #cv2.waitKey(0)

#сохранени
    '''
    save_path = os.path.join(OUTPUT_FOLDER)
    name = captcha_correct_text + ".jpg"
    all_path = os.path.join(save_path, name)
    cv2.imwrite(all_path, image_TH)
'''#нарезка
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

    #cv2.imshow("cropped", crop_img_TH)
    #cv2.waitKey(0)

    (x_short, y_short, w, h) = cv2.boundingRect(crop_img_TH)
    one_img_w = round(w/5)
    w_rezka_left = 0
    w_rezka_right = 30
    array_rezka_left  = []
    array_rezka_left.append(w_rezka_left)
    arrCaptcha_name = list(captcha_correct_text)
    for el in range(5):
        save_path = os.path.join(OUTPUT_FOLDER, arrCaptcha_name[el])

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        img_belia = np.full((50, 30), 255, np.uint8)
        one_img = crop_img_TH[0:h, el * one_img_w:(el+1)*one_img_w]
        for row in range(1, one_img.shape[0] - 1):
            for col in range(1, one_img.shape[1] - 1):
                img_belia[row, col] = one_img[row, col]

        p = os.path.join(save_path, "{}.png".format(str(counts_name).zfill(6)))
        cv2.imwrite(p, img_belia)
        #cv2.imshow("cropped", img_belia)
        #cv2.waitKey(0)
        counts_name = counts_name + 1
    '''
    for el in range(5):
        one_img_final = crop_img_TH[0:h, array_rezka_left[el]:array_rezka_left[el+1]]
        cv2.imshow("cropped", one_img_final)
        cv2.waitKey(0)

    for el in range(4):
        one_img = crop_img_TH[0:h, w_rezka_left:w_rezka_left+30]
        y_hith = 0
        y_hith_very_smal = 0
        y_hith_very_big = 0
        for row in range(1, one_img.shape[0] - 1):
            for col in range(round(one_img.shape[1] * 0.2),  round(one_img.shape[1] / 3)):
                if (one_img[row, col] == 0):
                    y_hith = row
                    break
            if (y_hith != 0):
                break
        hight_number = h - y_hith

        if (15 >= hight_number):
            for row in range(1, one_img.shape[0] - 1):
                for col in range(round((one_img.shape[1] * 2 )/ 3), one_img.shape[1] - round(one_img.shape[1] * 0.1)):
                    if (one_img[row, col] == 0):
                        y_hith_very_smal = row
                        break
                if (y_hith_very_smal != 0):
                    break
            hight_number = h - y_hith_very_smal

        if (45 >= hight_number > 40):
            for row in range(1, one_img.shape[0] - 1):
                for col in range(round((one_img.shape[1] * 2 )/ 3), one_img.shape[1] - round(one_img.shape[1] * 0.1)):
                    if (one_img[row, col] == 0):
                        y_hith_very_big = row
                        break
                if (y_hith_very_big != 0):
                    break

            hight_number = h - y_hith_very_big

        if(45 >= hight_number > 40):
            w_rezka_left = w_rezka_left + 30
        elif(40 >= hight_number > 35):
            w_rezka_left = w_rezka_left + 27
        elif (35 >= hight_number > 30):
            w_rezka_left = w_rezka_left + 23
        elif (30 >= hight_number > 15):
            w_rezka_left = w_rezka_left + 15

        w_rezka_right = w_rezka_left + 30
        array_rezka_left.append(w_rezka_left)
        #if(w_rezka_left <= w):
        #    cv2.imshow("cropped", one_img)
        #   cv2.waitKey(0)
    #if(w - array_rezka_left[4] > 30):
    #    array_rezka_left[4] = w - 30
    #    array_rezka_left.append(w)
    #else:
    array_rezka_left.append(w)
    print(array_rezka_left)

    for el in range(5):
        one_img_final = crop_img_TH[0:h, array_rezka_left[el]:array_rezka_left[el+1]]
        cv2.imshow("cropped", one_img_final)
        cv2.waitKey(0)

        if(15 >= hight_number):
            for row in range(1, one_img.shape[0] - 1):
                for col in range(one_img.shape[1] - 10, 24):
                    if (one_img[row, col] == 0):
                        y_hith_very_smal = row
                        break
                if (y_hith_very_smal != 0):
                    break

            hight_number = h - y_hith_very_smal

        if (45 >= hight_number > 40):
            for row in range(1, one_img.shape[0] - 1):
                for col in range(one_img.shape[1] - 10, 24):
                    if (one_img[row, col] == 0):
                        y_hith_very_big = row
                        break
                if (y_hith_very_big != 0):
                    break

            hight_number = h - y_hith_very_big

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