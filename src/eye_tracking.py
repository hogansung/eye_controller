from imutils.video import VideoStream
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
import cv2
import dlib
import os
import time
from constants import IMG_FILENAME


IMG_FOLDER = '../img/eyes_up_down'
THRESHOLD = 45

FILLED_GRAY_COLOR = 80

EYE_AR_LOWER_THRESH = 0.16
EYE_AR_EYEDN_THRESH = 0.19
EYE_AR_UPPER_THRESH = 0.26
EYE_AD_EYEUP_THRESH = 2.25
EYE_AD_EYELR_THRESH = 2.25
EYE_AR_CONSEC_FRAMES = 3


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
    # (
    #     lens_x, lens_y, lens_w, lens_h
    # ) = min_loc_x-lens_dia//2, min_loc_y-3*lens_dia//2, lens_dia, lens_dia

    # cv2.imshow('zz', extended_gray_roi)
    # cv2.waitKey(0)


    #gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    # cv2.imshow('zz', gray_roi)
    # cv2.waitKey(0)


    # circles = cv2.HoughCircles(gray_roi,cv2.HOUGH_GRADIENT,1,minDist=50,
    #                         param1=150,param2=3,minRadius=12,maxRadius=14)
    # print(circles.shape)
    # try:
    #     for cx, cy, cr in circles[0, :]:
    #         print(cx, cy, cr)
    #         # draw the outer circle
    #         cv2.circle(roi, (cx, cy), cr, (255, 255, 255), 1)
    #         print("drawing circle")
    #         # draw the center of the circle
    #         #cv2.circle(roi, (i[0], i[1]), 2, (255, 255, 255), 3)
    # except Exception as e:
    #     print(e)


    # _, threshold = cv2.threshold(gray_roi, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('zz', threshold)
    # cv2.waitKey(0)
    # cv2.imshow('zz', gray_roi)
    # cv2.waitKey(0)

    # contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if not contours:
    #     return None, None, None, None

    # for contour in contours:
    #     cv2.drawContours(roi, [contour], -1, (0, 0, 255), 3)
    # cv2.imshow('zz', roi)
    # cv2.waitKey(1)

    # contour = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[0]

    # lens_x, lens_y, lens_w, lens_h = cv2.boundingRect(contour)
    # cv2.drawContours(roi, [contour], -1, (0, 0, 255), 3)
    # cv2.rectangle(roi, (lens_x, lens_y), (lens_x + lens_w, lens_y + lens_h), (255, 0, 0), 1)
    # cv2.line(roi, (lens_x + int(lens_w / 2), 0), (lens_x + int(lens_w / 2), n_row), (0, 255, 0), 1)
    # cv2.line(roi, (0, lens_y + int(lens_h / 2)), (n_col, lens_y + int(lens_h / 2)), (0, 255, 0), 1)
    # cv2.imshow('zz', roi)
    # cv2.waitKey(0)

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


def cal_eye_aspect_ratio(eyes_xy_list):
    # compute the euclidean distances between two sets of vertical eye landmarks
    A = dist.euclidean(eyes_xy_list[1], eyes_xy_list[5])
    B = dist.euclidean(eyes_xy_list[2], eyes_xy_list[4])

    # compute the euclidean distance between the horizontal eye landmark
    C = dist.euclidean(eyes_xy_list[0], eyes_xy_list[3])

    # compute the eye aspect ratio
    ratio = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ratio


def parse_eyes_movement(
        l_fx,
        r_fx,
):
    l_ratio = cal_eye_aspect_ratio(l_fx)
    r_ratio = cal_eye_aspect_ratio(r_fx)
    ratio = (l_ratio + r_ratio) / 2.0
    print(ratio)

    # check if the eyes are closed or looking downward
    if ratio < EYE_AR_LOWER_THRESH:
        return 'I'
    elif ratio < EYE_AR_EYEDN_THRESH:
        return 'D'

    dx = np.mean([x for x, y in l_fx + r_fx])
    dy = np.mean([y for x, y in l_fx + r_fx])
    print(dx, dy)

    # check if eyes are moving upward
    if dy > EYE_AD_EYEUP_THRESH:
        return 'U'

    # check if eyes are moving left/right
    if abs(dx) > EYE_AD_EYELR_THRESH:
        if dx > 0:
            return 'R'
        else:
            return 'L'

    # check if eyes are enlarged
    if ratio > EYE_AR_UPPER_THRESH:
        return 'O'

    return 'N'


def parse_face(detector, predictor, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect face
    rects = detector(gray, 1)
    if not rects:
        print('No face is identified.')
        return None
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


    status = parse_eyes_movement(
        l_fx,
        r_fx,
    )
    print(status)

    # label left and right eyes
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
        if image is not None:
            cv2.imshow("Image", image)

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


def online_parsing(detector, predictor):
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(False).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        #frame = imutils.resize(frame, width=400)

        image = parse_face(detector, predictor, frame)
        if image is not None:
            cv2.imshow("Image", image)

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


def main():
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../dat/shape_predictor_68_face_landmarks.dat')

    # offline parsing
    # offline_parsing(
    #     detector=detector,
    #     predictor=predictor,
    # )

    # online parsing
    online_parsing(
        detector=detector,
        predictor=predictor,
    )

if __name__ == '__main__':
    main()