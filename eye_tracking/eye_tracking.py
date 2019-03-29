from imutils import face_utils

import numpy as np
import cv2


from constants.constants import *


def locate_lens(roi):
    n_row, n_col, _ = roi.shape
    lens_dia = int(n_col * 0.4)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    extended_gray_roi = cv2.copyMakeBorder(
        src=gray_roi,
        top=lens_dia,
        bottom=lens_dia,
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )

    for i in range(n_row + 2*lens_dia):
        for j in range(n_col):
            if extended_gray_roi[i][j] == 255:
                extended_gray_roi[i][j] = FILLED_GRAY_COLOR

    filtered_roi = cv2.filter2D(
        src=extended_gray_roi,
        ddepth=cv2.CV_64F,
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lens_dia, lens_dia)),
        anchor=(-1,-1),
    )

    min_val, _, (min_loc_x, min_loc_y), _ = cv2.minMaxLoc(filtered_roi)
    return min_loc_x, min_loc_y - lens_dia, lens_dia


def parse_eyes(image, eyes_xy_list):
    # find founding rect for each eeye
    eye_x, eye_y, eye_w, eye_h = cv2.boundingRect(eyes_xy_list)
    # roi = image[eye_y: eye_y+eye_h, eye_x: eye_x + eye_w]

    # masked each eye
    mask = 255 - np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, np.array([eyes_xy_list]), 0)
    masked_image = cv2.bitwise_or(image, mask)
    masked_roi = masked_image[eye_y: eye_y+eye_h, eye_x: eye_x + eye_w]

    # locate each eye
    lens_cx, lens_cy, lens_cr = locate_lens(masked_roi)

    return eye_x, eye_y, eye_w, eye_h, eye_x+lens_cx, eye_y+lens_cy, lens_cr//2


def label_lens(
        image,
        lens_cx,
        lens_cy,
        lens_cr,
        eyes_xy_list,
        color_lens,
        color_eyes,
        color_cross,
):
    n_row, n_col, _ = image.shape

    cv2.circle(
        img=image,
        center=(lens_cx, lens_cy),
        radius=lens_cr,
        color=color_lens,
        thickness=1,
    )
    cv2.polylines(
        img=image,
        pts=np.int32([eyes_xy_list]),
        isClosed=True,
        color=color_eyes,
        thickness=1,
    )
    cv2.line(
        img=image,
        pt1=(lens_cx, 0),
        pt2=(lens_cx, n_row),
        color=color_cross,
        thickness=1,
    )
    cv2.line(
        img=image,
        pt1=(0, lens_cy),
        pt2=(n_col, lens_cy),
        color=color_cross,
        thickness=1,
    )


# def cal_eye_aspect_ratio(eyes_xy_list):
#     # compute the euclidean distances between two sets of vertical eye landmarks
#     A = dist.euclidean(eyes_xy_list[1], eyes_xy_list[5])
#     B = dist.euclidean(eyes_xy_list[2], eyes_xy_list[4])
#
#     # compute the euclidean distance between the horizontal eye landmark
#     C = dist.euclidean(eyes_xy_list[0], eyes_xy_list[3])
#
#     # compute the eye aspect ratio
#     ratio = (A + B) / (2.0 * C)
#
#     # return the eye aspect ratio
#     return ratio


# def parse_eyes_movement(
#         l_fx,
#         r_fx,
# ):
#     l_ratio = cal_eye_aspect_ratio(l_fx)
#     r_ratio = cal_eye_aspect_ratio(r_fx)
#     ratio = (l_ratio + r_ratio) / 2.0
#     # print(ratio)
#
#     # check if the eyes are closed or looking downward
#     if ratio < EYE_AR_LOWER_THRESH:
#         return 'I'
#     elif ratio < EYE_AR_EYEDN_THRESH:
#         return 'D'
#
#     dx = np.mean([x for x, y in l_fx + r_fx])
#     dy = np.mean([y for x, y in l_fx + r_fx])
#     # print(dx, dy)
#
#     # check if eyes are moving upward
#     if dy > EYE_AD_EYEUP_THRESH:
#         return 'U'
#
#     # check if eyes are moving left/right
#     if abs(dx) > EYE_AD_EYELR_THRESH:
#         if dx > 0:
#             return 'R'
#         else:
#             return 'L'
#
#     # check if eyes are enlarged
#     if ratio > EYE_AR_UPPER_THRESH:
#         return 'O'
#
#     return 'N'


def parse_face(detector, predictor, image, label_switch=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect face
    rects = detector(gray, 1)
    if not rects:
        print('No face is identified.')
        return None, 'N'
    rect = rects[0]

    # detect different elements on face
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    l_eye_xy_list = np.array(shape[42:48])
    r_eye_xy_list = np.array(shape[36:42])

    # extract left and right eyes
    (
        l_eye_x, l_eye_y, l_eye_w, l_eye_h,
        l_lens_cx, l_lens_cy, l_lens_cr
    ) = parse_eyes(image, l_eye_xy_list)
    (
        r_eye_x, r_eye_y, r_eye_w, r_eye_h,
        r_lens_cx, r_lens_cy, r_lens_cr
    ) = parse_eyes(image, r_eye_xy_list)

    # reset origin at the center of lens
    l_fx = [(x - l_lens_cx, y - l_lens_cy) for x, y in l_eye_xy_list]
    r_fx = [(x - r_lens_cx, y - r_lens_cy) for x, y in r_eye_xy_list]


    # status = parse_eyes_movement(
    #     l_fx,
    #     r_fx,
    # )

    # label left and right eyes
    if label_switch:
        label_lens(
            image=image,
            lens_cx=l_lens_cx,
            lens_cy=l_lens_cy,
            lens_cr=l_lens_cr,
            eyes_xy_list=l_eye_xy_list,
            color_lens=(255,255,255),
            color_eyes=(0,255,0),
            color_cross=(255,0,0),
        )
        label_lens(
            image=image,
            lens_cx=r_lens_cx,
            lens_cy=r_lens_cy,
            lens_cr=r_lens_cr,
            eyes_xy_list=r_eye_xy_list,
            color_lens=(255, 255, 255),
            color_eyes=(0, 255, 0),
            color_cross=(0, 0, 255),
        )

    return image, l_fx, r_fx