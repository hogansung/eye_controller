import os
import pickle
import time

import cv2
import dlib

from calibration.calibration import calibrate, predict_action_types
from constants.action_types import ActionTypes
from constants.constants import *
from eye_tracking.eye_tracking import parse_face


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


def offline_parsing(model, username, detector, predictor):
    targets = os.listdir('img/{USERNAME}/'.format(USERNAME=username))
    for target in targets:
        target_image_folder_path = 'img/{USERNAME}/{TARGET}/'.format(
            USERNAME=username,
            TARGET=target,
        )

        num_files = len(os.listdir(target_image_folder_path))
        print(target, num_files)

        for idx in range(num_files)[:30]:
            print(idx)
            image_filename = 'img/{USERNAME}/{TARGET}/frame_{INDEX:04d}.jpg'.format(
                USERNAME=username,
                TARGET=target,
                INDEX=idx,
            )

            image = cv2.imread(image_filename)
            image, l_fx, r_fx = parse_face(detector, predictor, image, label_switch=True)
            if l_fx is None:
                continue

            data = [
                x for p in l_fx + r_fx for x in p
            ]
            status = ActionTypes(model.predict([data])[0])

            cv2.imshow('Image', image)
            key = cv2.waitKey(1)

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break


def online_parsing(model, detector, predictor):
    print("[INFO] camera sensor warming up...")
    cap = cv2.VideoCapture(0)

    # prepare the demo layout
    while True:
        ret, image = cap.read()

        # show image
        cv2.imshow('Image', image)

        # if the `g` key was pressed, break from the loop
        key = cv2.waitKey(1)
        if key == ord("g"):
            break

    # notify user the camera is ready
    print('\a', end='')
    for idx in range(COUNT_DOWN_SECS)[::-1]:
        print('Counting down: {CNT}'.format(CNT=idx))
        time.sleep(1.0)

    pre_status = ActionTypes.OTHERS
    count = 0
    while True:
        ret, image = cap.read()
        image, l_fx, r_fx = parse_face(detector, predictor, image, label_switch=True)
        if l_fx is None:
            continue

        data_x = [[
            x for p in l_fx + r_fx for x in p
        ]]
        status = predict_action_types(model, data_x)

        # logic of dealing with status
        if status != pre_status:
            pre_status = status
            count = 0

        if count == EYE_MOVEMENT_THRESH:
            execute_action(status)
            count = 1
        count += 1

        # print('There are {COUNT} continuous {ACTION}'.format(COUNT=count, ACTION=status))

        # show image
        cv2.imshow('Image', image)
        key = cv2.waitKey(1)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    print("[INFO] camera sensor closes.")
    cap.release()


def cleanup(username):
    data_folder_path = 'dat/{USERNAME}'.format(
        USERNAME=username,
    )
    image_folder_path = 'img/{USERNAME}'.format(
        USERNAME=username,
    )
    model_path = 'mdl/{USERNAME}.pickle'.format(USERNAME=username)
    try:
        # shutil.rmtree(data_folder_path)
        # shutil.rmtree(image_folder_path)
        os.remove(model_path)
    except OSError as e:
        pass


def main():
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    print(os.getcwd())
    predictor = dlib.shape_predictor('dat/shape_predictor_68_face_landmarks.dat')

    # ask the username
    username = input("Who are you?")
    print('Your username is {USERNAME}'.format(USERNAME=username))

    model_path = 'mdl/{USERNAME}.pickle'.format(USERNAME=username)
    # if model is cached:
    if os.path.isfile(model_path):
        while True:
            res = input("Do you want to re-calibrate your model? (Y/N)")
            if res in ('Y', 'y'):
                cleanup(username=username)
                model = calibrate(username, detector, predictor)
                pickle.dump(model, open(model_path, 'wb'))
                break
            elif res in ('N', 'n'):
                model = pickle.load(open(model_path, 'rb'))
                print('Your model is fetched from caches.')
                break
            else:
                print('Please only type in Y or N as your response.')
    else:
        print('Your model does not exist.')
        while True:
            res = input("Do you want to calibrate your model? (Y/N)")
            if res in ('Y', 'y'):
                cleanup(username=username)
                model = calibrate(username, detector, predictor)
                pickle.dump(model, open(model_path, 'wb'))
                break
            elif res in ('N', 'n'):
                model = pickle.load(open(GENERAL_MODEL_PATH, 'rb'))
                print('General model is fetched from caches.')
                break
            else:
                print('Please only type in Y or N as your response.')

    # # offline parsing
    # offline_parsing(
    #     model=model,
    #     username=username,
    #     detector=detector,
    #     predictor=predictor,
    # )

    # online parsing
    online_parsing(
        model=model,
        detector=detector,
        predictor=predictor,
    )


if __name__ == '__main__':
    main()
