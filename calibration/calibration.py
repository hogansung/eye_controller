import os
import time

import cv2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from constants.action_types import ActionTypes
from constants.constants import *
from eye_tracking.eye_tracking import cal_eye_aspect_ratio, parse_face
from utils.utils import load_data


def generate_data(username, detector, predictor):
    print('Start data generation.')

    data_folder_path = 'dat/{USERNAME}/'.format(USERNAME=username)

    # if data is cached:
    if os.path.isdir(data_folder_path):
        data_x, data_y = load_data(data_folder_path)
        print('Your calibration data is fetched from caches.')
        return data_x, data_y

    # else, start making data from scratch
    print('Your calibration data does not exist.')
    os.makedirs(data_folder_path)

    data_x = []
    data_y = []
    print("[INFO] camera sensor warming up...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

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
        else:  # start fetching images from scratch
            print('Your images for {TARGET} do not exist.')
            os.makedirs(target_image_folder_path)

            print('After {SECONDS} after beep sound, please do {TARGET}'.format(
                SECONDS=COUNT_DOWN_SECS,
                TARGET=target,
            ))

            # notify user that calibration is happening soon
            print('\a', end='')

            # count down for COUNT_DOWN_SECS
            for idx in range(COUNT_DOWN_SECS)[::-1]:
                print('Counting down: {COUNT}'.format(COUNT=idx))
                time.sleep(1.0)

            # start extracting frames from video stream
            images = []
            for idx in range(NUM_IMAGES):
                ret, image = cap.read()
                cv2.imwrite(
                    filename=os.path.join(
                        target_image_folder_path,
                        'frame_{INDEX:04d}.jpg'.format(INDEX=idx)
                    ),
                    img=image,
                )
                images.append(image)

        image_dict[target] = images

    print("[INFO] camera sensor closes.")
    cap.release()

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
            f.write(','.join(map(str, x)) + '\n')
    with open(os.path.join(data_folder_path, 'data_y.txt'), 'w') as f:
        for y in data_y:
            f.write(str(y) + '\n')

    print('End data generation.')
    return data_x, data_y


def train_test_split_in_sequence(data_x, data_y):
    n_data = len(data_x)
    assert n_data % NUM_IMAGES == 0

    tn_x, tt_x, tn_y, tt_y = [], [], [], []
    for i in range(0, n_data, NUM_IMAGES):
        tn_x += data_x[i:i + NUM_TN_IMAGES]
        tt_x += data_x[i + NUM_TN_IMAGES:i + NUM_IMAGES]
        tn_y += data_y[i:i + NUM_TN_IMAGES]
        tt_y += data_y[i + NUM_TN_IMAGES:i + NUM_IMAGES]

    return tn_x, tt_x, tn_y, tt_y


def build_model(data_x, data_y):
    # remove ActionTypes.ZOOM_IN in training, given that it is too noisy
    data_x, data_y = zip(*filter(
        lambda x: x[1] != ActionTypes.ZOOM_IN.value, zip(data_x, data_y)
    ))

    # split train and test data in sequence
    tn_x, tt_x, tn_y, tt_y = train_test_split_in_sequence(data_x=data_x, data_y=data_y)

    # build up a model
    # clf = LogisticRegression(
    #     random_state=514,
    #     solver='lbfgs',
    #     multi_class='multinomial',
    #     max_iter=5000,
    # )
    # clf = SVC(
    #     C=1,
    #     gamma='auto',
    #     decision_function_shape = 'ovr',
    #     random_state=514,
    # )
    clf = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_features=None,
        max_depth=3,
        random_state=514,
    )

    # validate model performance
    clf.fit(tn_x, tn_y)
    tn_p = clf.predict(tn_x)
    tt_p = clf.predict(tt_x)

    tn_acc = accuracy_score(tn_y, tn_p)
    print('ACC for TN data is {ACC}'.format(ACC=tn_acc))
    tt_acc = accuracy_score(tt_y, tt_p)
    for y, p in zip(tt_y, tt_p):
        if y != p:
            print(y, p)
    print('ACC for TT data is {ACC}'.format(ACC=tt_acc))

    # train on all data
    clf.fit(data_x, data_y)
    data_p = clf.predict(data_x)

    data_acc = accuracy_score(data_y, data_p)
    print('ACC for whole data is {ACC}'.format(ACC=data_acc))

    # return model
    return clf


def predict_action_types(model, data_x):
    if ActionTypes.ZOOM_IN.value in model.classes_:
        return ActionTypes(model.predict(data_x)[0])
    else:
        # test ActionTypes.ZOOM_IN separately, given that it is too noisy
        ratio = 0.5 * (
                cal_eye_aspect_ratio(data_x[0][:12]) + cal_eye_aspect_ratio(
            data_x[0][12:])
        )
        # print(ratio)
        if ratio < EYE_AR_LOWER_THRESH:
            return ActionTypes.ZOOM_IN
        else:
            return ActionTypes(model.predict(data_x)[0])


def calibrate(username, detector, predictor):
    print('Start calibration.')

    # generate a list of data
    data_x, data_y = generate_data(username, detector, predictor)

    # build a model
    model = build_model(data_x, data_y)

    print('End calibration.')
    return model
