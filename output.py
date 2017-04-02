"""
Caterpillar Tube Pricing

Nguyen Huy Anh, Lee Vicson, Deon Seng, Oh Yoke Chew
"""

import os
import numpy as np

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'competition_data')
# inputs
TRAIN_FILE = os.path.join(DATA_DIR, 'train_set.csv')
TUBE_FILE = os.path.join(DATA_DIR, 'tube.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_set.csv')
# outputs
OUT_FILE = os.path.join(CUR_DIR, 'out.csv')
EXTRACT_TRAIN_FILE = os.path.join(CUR_DIR, 'extracted_train.csv')
EXTRACT_TUBE_FILE = os.path.join(CUR_DIR, 'extracted_tube.csv')
MERGE_TRAIN_TUBE_FILE = os.path.join(CUR_DIR, 'merged_train_tube.csv')


def extract_train():
    """
    Extract train_set.csv.
    Only extract tubes with bracket pricing and quantity 1.
    """
    with open(TRAIN_FILE, 'r') as in_, open(EXTRACT_TRAIN_FILE, 'w') as out_:
        tmp = in_.readlines()
        head = tmp[0].strip().split(',')
        out_.write(head[0] + ',' + head[7] + '\n')
        for line in tmp[1:]:
            values = line.strip().split(',')
            if int(values[3]) == int(values[4]) == 0 and values[5] == 'Yes' and int(values[6]) == 1:
                out_.write(values[0] + ',' + values[7] + '\n')


def extract_tubes():
    """
    Extract tube.csv.
    Only extract tubes with no special characteristics.
    """
    with open(TUBE_FILE, 'r') as in_, open(EXTRACT_TUBE_FILE, 'w') as out_:
        tmp = in_.readlines()
        head_tmp = tmp[0].strip().split(',')
        head = [head_tmp[0]] + head_tmp[2:7]
        out_.write(','.join(head) + '\n')
        for line in tmp[1:]:
            values = line.strip().split(',')
            if int(values[-1]) == int(values[-2]) == int(values[-3]) == 0 and values[-6] == values[-7] == values[-8] == values[-9] == 'N':
                values_tmp = [values[0]] + values[2:7]
                out_.write(','.join(values_tmp) + '\n')


def merge_train_tubes():
    """
    Merge two data sets from extract_train and extract_tubes together.
    """
    with open(EXTRACT_TRAIN_FILE, 'r') as train_, open(EXTRACT_TUBE_FILE, 'r') as tube_, open(MERGE_TRAIN_TUBE_FILE, 'w') as out_:
        tmp_train = train_.readlines()
        tmp_tube = tube_.readlines()
        head_tmp = tmp_tube[0].strip().split(
            ',') + [tmp_train[0].strip().split(',')[1]]
        out_.write(','.join(head_tmp) + '\n')
        for tube_line in tmp_tube[1:]:
            tube_tmp = tube_line.strip().split(',')
            key = tube_tmp[0]
            for train_line in tmp_train[1:]:
                train_tmp = train_line.strip().split(',')
                if train_tmp[0] == key:
                    value_tmp = tube_tmp + [train_tmp[1]]
                    out_.write(','.join(value_tmp) + '\n')
                    break


def get_cost_coefficients_for_tube():
    """
    Get the coefficients for linear regression of tube cost.
    Model: cost = f(diameter, wall, length, num_bends, bend_radius) ::= y = Ax
    """
    y_vect = list()
    diameter_vect = list()
    wall_vect = list()
    length_vect = list()
    num_bends_vect = list()
    bend_radius_vect = list()
    with open(MERGE_TRAIN_TUBE_FILE, 'r') as merged_:
        tmp = merged_.readlines()[1:]
        for line in tmp:
            values = line.strip().split(',')
            y_vect.append(float(values[-1]))
            diameter_vect.append(float(values[1]))
            wall_vect.append(float(values[2]))
            length_vect.append(float(values[3]))
            num_bends_vect.append(float(values[4]))
            bend_radius_vect.append(float(values[5]))
    a_mat = [diameter_vect, wall_vect, length_vect,
             num_bends_vect, bend_radius_vect]
    a_mat_big = np.column_stack(a_mat + [[1] * len(a_mat[0])])
    x_vect = np.linalg.lstsq(a_mat_big, y_vect)[0]
    return x_vect


def predict():
    """
    Predict the price of each tube.
    """
    with open(TEST_FILE, 'r') as in_, open(OUT_FILE, 'w') as out_:
        ids = [line.strip().split(',')[0] for line in in_.readlines()[1:]]
        out_.write('id,cost\n')
        for id_ in ids:
            out_.write('{},1\n'.format(id_))

if __name__ == '__main__':
    get_cost_coefficients_for_tube()
    predict()
