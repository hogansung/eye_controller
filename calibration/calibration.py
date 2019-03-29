from eye_tracking.eye_tracking import parse_face
from imutils.video import VideoStream

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from constants.constants import *
from constants.action_types import ActionTypes

import cv2
import os
import time

from constants.constants import NUM_IMAGES


def generate_data(username, detector, predictor):
    print(os.getcwd())
    print('Start data generation.')

    data_folder_path = 'dat/{USERNAME}/'.format(
        USERNAME=username,
    )

    # if data is cached:
    if os.path.isdir(data_folder_path):
        data_x = [
            list(map(float, line.strip().split(','))) for line in open(
                os.path.join(data_folder_path, 'data_x.txt')
            ).readlines()
        ]
        data_y = [
            list(map(float, line.strip())) for line in open(
                os.path.join(data_folder_path, 'data_y.txt')
            ).readlines()
        ]
        print('Your calibration data is fetched from caches.')
        return data_x, data_y

    # else, start making data from scratch
    print('Your calibration data does not exist.')
    os.makedirs(data_folder_path)

    data_x = []
    data_y = []
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(False).start()

    targets = [
        ActionTypes.OTHERS, ActionTypes.UP, ActionTypes.DOWN,
        ActionTypes.LEFT, ActionTypes.RIGHT, ActionTypes.ZOOM_IN, ActionTypes.ZOOM_OUT,
    ]

    image_dict = dict()
    for target in targets:
        target_image_folder_path = 'img/{USERNAME}/{TARGET}/'.format(
            USERNAME=username,
            TARGET=target,
        )

        # if images are cached for certain target:
        if os.path.isdir(target_image_folder_path):
            images = [
                cv2.imread(
                    os.path.join(target_image_folder_path, f)
                ) for f in os.listdir(target_image_folder_path)
            ]
            assert NUM_IMAGES == len(images)
            print('Your images for {TARGET} is fetched from caches.')
        else: # start fetching images from scratch
            print('Your images for {TARGET} do not exist.')
            os.makedirs(target_image_folder_path)

            print('After {SECONDS} after beep sound, please do {TARGET}'.format(
                SECONDS=COUNT_DOWN_TIMES,
                TARGET=target,
            ))

            # notify user that calibration is happening soon
            print('\a')

            # count down for COUNT_DOWN_TIMES
            for idx in range(COUNT_DOWN_TIMES)[::-1]:
                print('Counting down: {COUNT}'.format(COUNT=idx))
                time.sleep(1.0)

            # start extracting frames from video stream
            images = []
            for idx in range(NUM_IMAGES):
                image = vs.read()
                cv2.imwrite(
                    filename=os.path.join(
                        target_image_folder_path,
                        'frame_{INDEX:04d}.jpg'.format(INDEX=idx)
                    ),
                    img=image,
                )
                images.append(image)

        image_dict[target] = images

    print("[INFO] camera sensor closes")
    vs.stop()

    # start parsing images
    print('Start parsing images.')
    for target in targets:
        print('Now is working on {TARGET}'.format(TARGET=target))
        for image in image_dict[target]:
            _, l_fx, r_fx = parse_face(detector, predictor, image)
            data_x.append([
                x for p in l_fx + r_fx for x in p
            ])
            data_y.append(target.value)
            print('.', end='', flush=True)
        print('')
    print('End parsing images.')

    with open(os.path.join(data_folder_path, 'data_x.txt'), 'w') as f:
        for x in data_x:
            f.write(','.join(map(str, x))+ '\n')
    with open(os.path.join(data_folder_path, 'data_y.txt'), 'w') as f:
        for y in data_y:
            f.write(str(y) + '\n')

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
    tn_acc = accuracy_score(tn_y, tn_p)
    print('ACC for TN data is {RMSE}'.format(RMSE=tn_acc))
    tt_acc = accuracy_score(tt_y, tt_p)
    print('ACC for TT data is {RMSE}'.format(RMSE=tt_acc))

    print('End calibration.')
    return clf