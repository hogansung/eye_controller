from imutils.video import VideoStream

from calibration.calibration import calibrate
from constants.constants import *
from constants.action_types import ActionTypes
from eye_tracking.eye_tracking import parse_face

import cv2
import dlib
import os
import pickle
import time


def execute_action(action):
    print('Execute action: {ACTION}'.format(ACTION=action))
    if ActionTypes.UP == action:
        pg.drag(0, +MOUSE_DRAG_DISTANCE, MOUSE_DRAG_DURATION, button='left')
    elif ActionTypes.DOWN == action:
        pg.drag(0, -MOUSE_DRAG_DISTANCE, MOUSE_DRAG_DURATION, button='left')
    elif ActionTypes.LEFT == action:
        pg.drag(+MOUSE_DRAG_DISTANCE, 0, MOUSE_DRAG_DURATION, button='left')
    elif ActionTypes.RIGHT == action:
        pg.drag(-MOUSE_DRAG_DISTANCE, 0, MOUSE_DRAG_DURATION, button='left')
    elif ActionTypes.ZOOM_IN == action:
        pg.scroll(+MOUSE_SCROLL_DISTANCE)
    elif ActionTypes.ZOOM_OUT == action:
        pg.scroll(-MOUSE_SCROLL_DISTANCE)
    pg.moveTo(SCREEN_W / 2, SCREEN_H / 2, 0)


# def offline_parsing(detector, predictor):
#     num_files = len(os.listdir(IMG_FOLDER))
#     print(num_files)
#
#     for idx in range(num_files):
#         print(idx)
#         img_filename = IMG_FILENAME.format(
#             IMG_FOLDER=IMG_FOLDER,
#             INDEX=idx,
#         )
#
#         image = cv2.imread(img_filename)
#         image, status = parse_face(detector, predictor, image)
#         key = cv2.imshow('Image', image)
#
#         # if the `q` key was pressed, break from the loop
#         if key == ord("q"):
#             break

def online_parsing(model, detector, predictor):
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(False).start()

    for idx in range(COUNT_DOWN_TIMES)[::-1]:
        print('Counting down: {CNT}'.format(CNT=idx))
        time.sleep(1.0)

    pre_status = ActionTypes.OTHERS
    count = 0
    while True:
        image = vs.read()
        image, l_fx, r_fx = parse_face(detector, predictor, image)

        status = ActionTypes(model.predict(l_fx + r_fx))

        # logic of dealing with status
        if status != pre_status:
            pre_status = status
            count = 0

        if count == EYE_MOVEMENT_THRESH:
            execute_action(status)
            count = 1
        count += 1

        print('There are {COUNT} continuous {ACTION}'.format(COUNT=count, ACTION=status))

        # if the `q` key was pressed, break from the loop
        key = cv2.imshow('Image', image)
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

    # ask the username
    username = 'hogan'
    print('Your username is {USERNAME}'.format(USERNAME=username))

    model_path = 'mdl/{USERNAME}.pickle'.format(USERNAME=username)
    # if model is cached:
    if os.path.isfile(model_path):
        model = pickle.load(open('model_path', 'rb'))
        print('Your model is fetched from caches.')
    else:
        print('Your model does not exist.')
        model = calibrate(username, detector, predictor)
        pickle.dump(model, open(model_path, 'wb'))

    # online parsing
    online_parsing(
        model=model,
        detector=detector,
        predictor=predictor,
    )

if __name__ == '__main__':
    main()