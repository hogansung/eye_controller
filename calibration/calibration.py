from eye_tracking.eye_tracking import parse_face
from imutils.video import VideoStream

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from constants.constants import *
from constants.action_types import ActionTypes

import cv2
import math
import os
import time

from constants.constants import NUM_IMAGES


def generate_data(username, detector, predictor):
    print('Start data generation.')

    data_folder_path = 'dat/{USERNAME}/'.format(
        USERNAME=username,
    )

    # if data is cached:
    if os.path.isdir(data_folder_path):
        data_x = [
            map(float, line.strip().split()) for line in open(
                os.path.join(data_folder_path, 'data_x.txt')
            ).readlines()
        ]
        data_y = [
            map(float, line.strip().split()) for line in open(
                os.path.join(data_folder_path, 'data_x.txt')
            ).readlines()
        ]
        print('Your calibration data is fetched from caches.')
        return data_x, data_y

    # else, start making data from scratch
    print('Your calibration data does not exist.')

    data_x = []
    data_y = []
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(False).start()

    targets = [
        ActionTypes.UP, ActionTypes.DOWN, ActionTypes.LEFT, ActionTypes.RIGHT,
        ActionTypes.ZOOM_IN, ActionTypes.ZOOM_OUT, ActionTypes.OTHERS,
    ]

    image_dict = dict()
    for target in targets:
        image_folder_path = 'img/{USERNAME}/{TARGET}/'.format(
            USERNAME=username,
            TARGET=target,
        )

        # if images are cached for certain target:
        if os.path.isdir(image_folder_path):
            images = [
                cv2.imread(
                    os.path.join(image_folder_path, f)
                ) for f in os.listdir(image_folder_path)
            ]
            assert NUM_IMAGES == len(images)
            print('Your images for {TARGET} is fetched from caches.')
        else: # start fetching images from scratch
            print('Your images for {TARGET} do not exist.')

            print('After {SECONDS} after beep sound, please do {TARGET}'.format(
                SECONDS=COUNT_DOWN_TIMES,
                TARGET=target,
            ))

            # notify user
            print('\a')

            # count down
            for idx in range(COUNT_DOWN_TIMES)[::-1]:
                print('Counting down: {CNT}'.format(CNT=idx))
                time.sleep(1.0)

            # start extracting frames from video stream
            images = []
            for idx in range(NUM_IMAGES):
                image = vs.read()
                cv2.imwrite(
                    filename=os.path.join(
                        image_folder_path,
                        'frame_{INDEX:04d}.jpg'.format(INDEX=idx)
                    ),
                    img=image,
                )
                images.append(image)

        image_dict[target] = images

    for target in targets:
        for image in image_dict[target]:
            _, l_fx, r_fx = parse_face(detector, predictor, image)
            data_x.append(l_fx + r_fx)
            data_y.append(target.value)

    print('End data generation.')
    return data_x, data_y


def calibrate(username, detector, predictor):
    print('Start calibration.')

    # generate a list of data
    data_x, data_y = generate_data(username, detector, predictor)

    # split train and test data with random shuffle
    tn_x, tt_x, tn_y, tt_y = train_test_split(
        data_x,
        data_y,
        test_size=0.33,
        random_state=514,
    )

    # build up a model
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    clf.fit(tn_x, tn_y)

    # make predictions
    tn_p = clf.predict(tn_x)
    tt_p = clf.predict(tt_x)

    # evaluate model performance
    tn_rmse = math.sqrt(mean_squared_error(tn_y, tn_p))
    print('RMSE for TN data is {RMSE}'.format(RMSE=tn_rmse))
    tt_rmse = math.sqrt(mean_squared_error(tt_y, tt_p))
    print('RMSE for TT data is {RMSE}'.format(RMSE=tt_rmse))

    print('End calibration.')

    return clf