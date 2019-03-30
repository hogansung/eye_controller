import os
import pickle

from calibration.calibration import build_model
from constants.constants import GENERAL_MODEL_PATH
from utils.utils import load_data


def main():
    usernames = [f for f in os.listdir('dat/') if os.path.isdir(os.path.join('dat/', f))]

    # get data from all available users
    data_x = []
    data_y = []
    for username in usernames:
        print('Now is parsing data for {USERNAME}'.format(USERNAME=username))
        data_folder_path = 'dat/{USERNAME}/'.format(USERNAME=username)
        _data_x, _data_y = load_data(data_folder_path)
        data_x += _data_x
        data_y += _data_y

    # build a general model
    print(data_y)
    print(len(data_y))
    model = build_model(data_x, data_y)

    # save the general model
    pickle.dump(model, open(GENERAL_MODEL_PATH, 'wb'))


if __name__ == '__main__':
    main()
