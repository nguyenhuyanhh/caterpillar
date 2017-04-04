"""
Caterpillar Tube Pricing

Nguyen Huy Anh, Lee Vicson, Deon Seng, Oh Yoke Chew
"""

import os
import numpy as np

# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'competition_data')
# inputs
TRAIN_FILE = os.path.join(DATA_DIR, 'train_set.csv')
TUBE_FILE = os.path.join(DATA_DIR, 'tube.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_set.csv')
# outputs
OUT_FILE = os.path.join(CUR_DIR, 'out.csv')
EXTRACT_TRAIN_BRACKET_FILE = os.path.join(
    CUR_DIR, 'extracted_train_bracket.csv')
EXTRACT_TRAIN_NO_BRACKET_FILE = os.path.join(
    CUR_DIR, 'extracted_train_no_bracket.csv')
MERGE_TRAIN_BRACKET_TUBE_FILE = os.path.join(
    CUR_DIR, 'merged_train_bracket_tube.csv')
MERGE_TRAIN_NO_BRACKET_TUBE_FILE = os.path.join(
    CUR_DIR, 'merged_train_no_bracket_tube.csv')


def extract_train(out_file, bracket='Yes', quantity='all'):
    """
    Extract train_set.csv.

    Attributes:
        bracket: str('Yes'/'No') - whether bracket pricing is applied
        quantity: int/str('all') - int to match quantity, 'all' to match all
        out_file: str - path to output file
    CSV header: tube_assembly_id,quantity,cost
    """
    tmp = list()
    with open(TRAIN_FILE, 'r') as in_:
        tmp = in_.readlines()
    with open(out_file, 'w') as out_:
        head = tmp[0].strip().split(',')
        head_tmp = [head[0]] + head[-2:]
        out_.write(','.join(head_tmp) + '\n')
        for line in tmp[1:]:
            values = line.strip().split(',')
            no_special = (int(values[3]) == int(values[4]) == 0)
            match = (values[5] == bracket and no_special) if quantity == 'all' else (
                values[5] == bracket and int(values[-2]) == quantity and no_special)
            if match:
                value_tmp = [values[0]] + values[-2:]
                out_.write(','.join(value_tmp) + '\n')


def extract_tube(out_file):
    """
    Extract tube.csv.

    Attributes:
        out_file: str - path to output file
    CSV header: tube_assembly_id,diameter,wall,length,num_bends,bend_radius
    """
    tmp = list()
    with open(TUBE_FILE, 'r') as in_:
        tmp = in_.readlines()
    with open(out_file, 'w') as out_:
        head = tmp[0].strip().split(',')
        head_tmp = [head[0]] + head[2:7]
        out_.write(','.join(head_tmp) + '\n')
        for line in tmp[1:]:
            values = line.strip().split(',')
            no_special = (int(values[-1]) == int(values[-2]) == int(values[-3]) ==
                          0 and values[-6] == values[-7] == values[-8] == values[-9] == 'N')
            if no_special:
                values_tmp = [values[0]] + values[2:7]
                out_.write(','.join(values_tmp) + '\n')


def merge_train_tube(in_train_file, in_tube_file, out_file):
    """
    Merge two data sets from extract_train and extract_tube together.

    Attributes:
        in_train_file: str - path to input train file
        in_tube_file: str - path to input tube file
        out_file: str - path to output file
    CSV header: tube_assembly_id,diameter,wall,length,num_bends,bend_radius,quantity,cost
    """
    tmp_train = list()
    tmp_tube = list()
    with open(in_train_file, 'r') as train_:
        tmp_train = train_.readlines()
    with open(in_tube_file, 'r') as tube_:
        tmp_tube = tube_.readlines()
    with open(out_file, 'w') as out_:
        head_tmp = tmp_tube[0].strip().split(
            ',') + tmp_train[0].strip().split(',')[1:]
        out_.write(','.join(head_tmp) + '\n')
        tr_ = 1
        tu_ = 1
        while tr_ < len(tmp_train) and tu_ < len(tmp_tube):
            train_tmp = tmp_train[tr_].strip().split(',')
            tube_tmp = tmp_tube[tu_].strip().split(',')
            if train_tmp[0] < tube_tmp[0]:
                tr_ += 1
                continue
            elif train_tmp[0] > tube_tmp[0]:
                tu_ += 1
                continue
            else:
                value_tmp = tube_tmp + train_tmp[1:]
                out_.write(','.join(value_tmp) + '\n')
                tr_ += 1
                tu_ += 1
                continue


def get_cost_coefficients_for_tube(in_file):
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
    with open(in_file, 'r') as merged_:
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


def get_model_tube_base_cost():
    """
    Get the model for tube base cost.
    """
    model_dir = os.path.join(CUR_DIR, 'model_tube_base_cost')
    extract_train_file = os.path.join(model_dir, 'extracted_train.csv')
    extract_tube_file = os.path.join(model_dir, 'extracted_tube.csv')
    merge_file = os.path.join(model_dir, 'merged.csv')

    if not os.path.exists(extract_train_file):
        extract_train(extract_train_file, quantity=1)
    if not os.path.exists(extract_tube_file):
        extract_tube(extract_tube_file)
    merge_train_tube(extract_train_file, extract_tube_file, merge_file)


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
    get_model_tube_base_cost()
