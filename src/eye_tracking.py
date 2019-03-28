from imutils import face_utils
import numpy as np
import imutils
import dlib
import os
import cv2
from constants import IMG_FILENAME


IMG_FOLDER = '../img/eye_rollings'
THRESHOLD = 50


def locate_eye(roi):
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    _, threshold = cv2.threshold(gray_roi, THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours)

    # for contour in contours:
    #     cv2.drawContours(roi, [contour], -1, (0, 0, 255), 3)
    # cv2.imshow('zz', roi)
    # cv2.waitKey(1)

    contour = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[0]

    lens_x, lens_y, lens_w, lens_h = cv2.boundingRect(contour)
    # cv2.drawContours(roi, [contour], -1, (0, 0, 255), 3)
    # cv2.rectangle(roi, (lens_x, lens_y), (lens_x + lens_w, lens_y + lens_h), (255, 0, 0), 1)
    # cv2.line(roi, (lens_x + int(lens_w / 2), 0), (lens_x + int(lens_w / 2), rows), (0, 255, 0), 1)
    # cv2.line(roi, (0, lens_y + int(lens_h / 2)), (cols, lens_y + int(lens_h / 2)), (0, 255, 0), 1)
    # cv2.imshow('zz', roi)
    # cv2.waitKey(1)

    return lens_x, lens_y, lens_w, lens_h


def parse_eye(image, xy_list):
    # find founding rect for each eeye
    eye_x, eye_y, eye_w, eye_h = cv2.boundingRect(np.array([xy_list]))

    # masked each eye
    mask = 255 - np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, np.array([xy_list]), 0)
    masked_image = cv2.bitwise_or(image, mask)
    masked_roi = masked_image[eye_y: eye_y+eye_h, eye_x: eye_x + eye_w]

    # locate each eye
    lens_x, lens_y, lens_w, lens_h = locate_eye(masked_roi)
    return eye_x, eye_y, eye_w, eye_h, lens_x, lens_y, lens_w, lens_h


def label_eye(
        image,
        eye_x, eye_y, eye_w, eye_h,
        lens_x, lens_y, lens_w, lens_h,
        color_square,
        color_cross,
):
    nrow, ncol, _ = image.shape
    cv2.rectangle(
        image,
        (eye_x+lens_x, eye_y+lens_y),
        (eye_x+lens_x+lens_w, eye_y+lens_y+lens_h),
        color_square,
        1,
    )
    cv2.line(
        image,
        (eye_x+lens_x + int(lens_w / 2), 0),
        (eye_x+lens_x + int(lens_w / 2), nrow),
        color_cross,
        1,
    )
    cv2.line(
        image,
        (0, eye_y+lens_y + int(lens_h / 2)),
        (ncol, eye_y+lens_y + int(lens_h / 2)),
        color_cross,
        1,
    )


def parse_eyes_movement(
        l_lens_mx, l_lens_my, l_eye_mx, l_eye_my, l_eye_w, l_eye_h,
        r_lens_mx, r_lens_my, r_eye_mx, r_eye_my, r_eye_w, r_eye_h,
):
    # check if the eyes are open or closed

    # check if the eyes are moving
    l_dx = l_lens_mx - l_eye_mx
    r_dx = r_lens_mx - r_eye_mx
    l_dy = l_lens_my - l_eye_my
    r_dy = r_lens_my - r_eye_my
    dx = (l_dx + r_dx) / 2
    dy = (l_dy + r_dy) / 2

    print(dx, dy)

    if abs(dx) > abs(dy):
        if dx > 0:
            return 'R'
        else:
            return 'L'
    else:
        if dy > 0:
            return 'D'
        else:
            return 'U'


def parse_face(detector, predictor, image):
    # resize image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect face
    rects = detector(gray, 1)
    assert len(rects) == 1
    rect = rects[0]

    # detect different elements on face
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # extract left and right eyes
    (
        r_eye_x, r_eye_y, r_eye_w, r_eye_h,
        r_lens_x, r_lens_y, r_lens_w, r_lens_h,
    ) = parse_eye(image, shape[36:42])
    (
        l_eye_x, l_eye_y, l_eye_w, l_eye_h,
        l_lens_x, l_lens_y, l_lens_w, l_lens_h,
    ) = parse_eye(image, shape[42:48])

    # make decision based on eyes movement
    l_lens_mx = l_lens_x + l_lens_w / 2
    l_lens_my = l_lens_y + l_lens_h / 2
    l_eye_mx = l_eye_w / 2
    l_eye_my = l_eye_h / 2
    r_lens_mx = r_lens_x + r_lens_w / 2
    r_lens_my = r_lens_y + r_lens_h / 2
    r_eye_mx = r_eye_w / 2
    r_eye_my = r_eye_h / 2

    status = parse_eyes_movement(
        l_lens_mx, l_lens_my, l_eye_mx, l_eye_my, l_eye_w, l_eye_h,
        r_lens_mx, r_lens_my, r_eye_mx, r_eye_my, r_eye_w, r_eye_h,
    )
    print(status)

    # label left and right eyes
    label_eye(
        image,
        r_eye_x, r_eye_y, r_eye_w, r_eye_h,
        r_lens_x, r_lens_y, r_lens_w, r_lens_h,
        (255, 0, 0),
        (0, 255, 0),
    )
    label_eye(
        image,
        l_eye_x, l_eye_y, l_eye_w, l_eye_h,
        l_lens_x, l_lens_y, l_lens_w, l_lens_h,
        (255, 0, 0),
        (0, 0, 255),
    )
    
    return image


def offline_parsing(detector, predictor):
    num_files = len(os.listdir(IMG_FOLDER))
    print(num_files)

    for idx in range(num_files):
        print(idx)
        img_filename = IMG_FILENAME.format(
            IMG_FOLDER=IMG_FOLDER,
            INDEX=idx,
        )

        image = cv2.imread(img_filename)
        image = parse_face(detector, predictor, image)
        cv2.imshow("Image", image)
        cv2.waitKey(1)


def online_parsing(detector, predictor):
    pass


def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../dat/shape_predictor_68_face_landmarks.dat')

    # offline parsing
    offline_parsing(
        detector=detector,
        predictor=predictor,
    )

    # online parsing
    online_parsing(
        detector=detector,
        predictor=predictor,
    )

if __name__ == '__main__':
    main()