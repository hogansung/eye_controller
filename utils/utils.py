import os


def load_data(data_folder_path):
    data_x = [
        list(map(float, line.strip().split(','))) for line in open(
            os.path.join(data_folder_path, 'data_x.txt')
        ).readlines()
    ]
    data_y = [
        float(line.strip()) for line in open(
            os.path.join(data_folder_path, 'data_y.txt')
        ).readlines()
    ]
    return data_x, data_y
