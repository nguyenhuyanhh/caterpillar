"""
Caterpillar Tube Pricing
v2: XGBoost

Nguyen Huy Anh, Lee Vicson, Deon Seng, Oh Yoke Chew
"""

from __future__ import print_function

import math
import os

import numpy as np
import xgboost as xgb

# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'competition_data')
MODEL_DIR = os.path.join(CUR_DIR, 'model_xgboost')
# inputs
TRAIN_FILE = os.path.join(DATA_DIR, 'train_set.csv')
TUBE_FILE = os.path.join(DATA_DIR, 'tube_mod.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_set.csv')
# constants
SUPP_ENCODE = ['S-0066', 'S-0041', 'S-0072',
               'S-0054', 'S-0026', 'S-0013', 'S-others']
# outputs
OUT_FILE = os.path.join(CUR_DIR, 'out.csv')


def preprocess_train(out_file):
    """
    Preprocess train_set.csv, with one-hot encoding for supplier and bracket

    Arguments:
        out_file: str - path to output file
    CSV header:
        tube_assembly_id,S-0066,S-0041,S-0072,S-0054,S-0026,S-0013,S-others,
        annual_usage,min_order_quantity,bracket_pricing,quantity,cost
    """
    tmp = list()
    with open(TRAIN_FILE, 'r') as in_:
        tmp = in_.readlines()
    with open(out_file, 'w') as out_:
        head = tmp[0].strip().split(',')
        head_tmp = [head[0]] + SUPP_ENCODE + head[-5:]
        out_.write(','.join(head_tmp) + '\n')
        for line in tmp[1:]:
            values = line.strip().split(',')
            # supplier
            encoding = ['0', '0', '0', '0', '0', '0', '0']
            if values[1] in SUPP_ENCODE:
                index = SUPP_ENCODE.index(values[1])
                encoding[index] = '1'
            else:
                encoding[-1] = '1'
            # bracket
            bracket = '1'
            if values[-3] == 'No':
                bracket = '0'
            value_tmp = [values[0]] + encoding + \
                values[-5:-3] + [bracket] + values[-2:]
            out_.write(','.join(value_tmp) + '\n')


def preprocess_test(out_file):
    """
    Preprocess test_set.csv, with one-hot encoding for supplier and bracket

    Arguments:
        out_file: str - path to output file
    CSV header:
        tube_assembly_id,S-0066,S-0041,S-0072,S-0054,S-0026,S-0013,S-others,
        annual_usage,min_order_quantity,bracket_pricing,quantity
    """
    tmp = list()
    with open(TEST_FILE, 'r') as in_:
        tmp = in_.readlines()
    with open(out_file, 'w') as out_:
        head = tmp[0].strip().split(',')
        head_tmp = [head[1]] + SUPP_ENCODE + head[-4:]
        out_.write(','.join(head_tmp) + '\n')
        for line in tmp[1:]:
            values = line.strip().split(',')
            # supplier
            encoding = ['0', '0', '0', '0', '0', '0', '0']
            if values[2] in SUPP_ENCODE:
                index = SUPP_ENCODE.index(values[2])
                encoding[index] = '1'
            else:
                encoding[-1] = '1'
            # bracket
            bracket = '1'
            if values[-2] == 'No':
                bracket = '0'
            value_tmp = [values[1]] + encoding + \
                values[-4:-2] + [bracket] + values[-1:]
            out_.write(','.join(value_tmp) + '\n')


def extract_tube(out_file):
    """
    Extract tube.csv.

    Arguments:
        out_file: str - path to output file
    CSV header:
        tube_assembly_id,diameter,wall,length,num_bends,bend_radius
    """
    tmp = list()
    total_weight = list()
    with open(TUBE_FILE, 'r') as in_:
        tmp = in_.readlines()
    with open('tube_total_weight.csv', 'r') as weight_:
        total_weight = weight_.readlines()
    with open(out_file, 'w') as out_:
        head = tmp[0].strip().split(',')
        head2 = total_weight[0].strip().split(',')
        head3 = ['weighted', 'not weighted']
        head_tmp = [head[0]] + head[2:13] + head2[1:-1] + head3 + [head2[-1]]
        out_.write(','.join(head_tmp) + '\n')
        i = 1
        for line in tmp[1:]:
            weight_raw = total_weight[i].strip().split(',')
            weight = [weight_raw[-1]]
            encoding = ['0', '0']
            if weight[0] == '0':
                encoding[1] = '1'
            else:
                encoding[0] = '1'
            values = line.strip().split(',')
            values_tmp = [values[0]] + values[2:13] + \
                weight_raw[1:-1] + encoding + weight
            out_.write(','.join(values_tmp) + '\n')
            i = i + 1


def merge_train_tube(in_train_file, in_tube_file, out_file):
    """
    Merge two data sets from extract_train and extract_tube together.

    Arguments:
        in_train_file: str - path to input train file
        in_tube_file: str - path to input tube file
        out_file: str - path to output file
    CSV header:
        tube_assembly_id,diameter,wall,length,num_bends,bend_radius,
        S-0066,S-0041,S-0072,S-0054,S-0026,S-0013,others,
        annual_usage,min_order_quantity,bracket_pricing,quantity,cost
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
                continue


def merge_test_tube():
    """
    Merge test_set.csv and tube.csv
    """
    with open(TEST_FILE, 'r') as in_:
        tmp_test = in_.readlines()[1:]
    with open('out_tube.csv', 'r') as tube_:
        tmp_tube = tube_.readlines()
    with open(os.path.join(CUR_DIR, 'out_test.csv'), 'w') as out_:
        for line in tmp_test:
            tmp = line.strip().split(',')
            tube_id = int(tmp[1][-5:])
            encoding = ['0', '0', '0', '0', '0', '0', '0']
            if tmp[2] in SUPP_ENCODE:
                index = SUPP_ENCODE.index(tmp[2])
                encoding[index] = '1'
            else:
                encoding[-1] = '1'
            if tube_id > 19490:
                tube_id = tube_id - 1
            tube_info = ((tmp_tube[tube_id]).strip().split(','))[1:]
            out_tmp = [tmp[1]] + tube_info + encoding + tmp[-4:]
            out_.write(','.join(out_tmp) + '\n')


def preprocess():
    """
    Wrapper for preprocessing functions.
    """
    pre_train_path = os.path.join(MODEL_DIR, 'pre_train.csv')
    pre_test_path = os.path.join(MODEL_DIR, 'pre_test.csv')

    preprocess_train(pre_train_path)
    preprocess_test(pre_test_path)


def train(features, output_model=True):
    """
    Build the model for prediction.

    Arguments:
        features: list(str) - list of features used to build the model
                features must match a header item in csv
        output_model: boolean - whether to output the model
                default is True. if False, output cv score only
    """
    # get training matrix
    lines = list()
    with open(os.path.join(MODEL_DIR, 'out_train_merged.csv'), 'r') as merged_:
        lines = merged_.readlines()
    vectors = lines[0].strip().split(',')
    no_vects = len(vectors)
    vects_lookup = {vectors[i]: i for i in range(no_vects)}
    vects = {i: list() for i in range(no_vects)}
    for line in lines[1:]:
        values = line.strip().split(',')
        for i in range(no_vects):
            if i == 0:  # id
                vects[i].append(int(values[i][-5:]))
            elif i == 25 or i == no_vects - 1:  # weight, cost
                vects[i].append(math.log10(float(values[i]) + 1))
            elif i == 35:  # bracket_pricing
                if values[i] == 'Yes':
                    vects[i].append(1)
                else:
                    vects[i].append(0)
            else:
                vects[i].append(float(values[i]))
    a_mat = list()
    for feat in features:
        if feat in vects_lookup.keys():
            a_mat.append(vects[vects_lookup[feat]])
    a_mat_big = np.column_stack(a_mat)

    # xgboost parameters
    dtrain = xgb.DMatrix(a_mat_big, label=vects[len(vects) - 1])
    param = {'max_depth': 8,
             'eta': 0.3,
             'min_child_weight': 5}
    num_round = 30

    # output model
    if output_model:
        model = xgb.train(param, dtrain, num_round)
        model.save_model(os.path.join(MODEL_DIR, '0001.model'))
    else:
        # using the built in cv method to check errors, it uses rmse though
        xgb.cv(param, dtrain, num_round, nfold=5, metrics={'rmse'}, seed=0, callbacks=[
            xgb.callback.print_evaluation(show_stdv=True)])


def predict(features):
    """
    Predict based on the model.

    Arguments:
        features: list(str) - list of features used to build the model
                features must match a header item in csv
    """
    # get test matrix
    lines = list()
    with open(os.path.join(MODEL_DIR, 'out_test.csv'), 'r') as merged_:
        lines = merged_.readlines()
    vectors = lines[0].strip().split(',')
    no_vects = len(vectors)
    vects_lookup = {vectors[i]: i for i in range(no_vects)}
    vects = {i: list() for i in range(no_vects)}
    for line in lines[1:]:
        values = line.strip().split(',')
        for i in range(no_vects):
            if i == 0:  # id
                vects[i].append(int(values[i][-5:]))
            elif i == 25:  # weight
                vects[i].append(math.log10(float(values[i]) + 1))
            elif i == 35:  # bracket_pricing
                if values[i] == 'Yes':
                    vects[i].append(1)
                else:
                    vects[i].append(0)
            else:
                vects[i].append(float(values[i]))
    a_mat = list()
    for feat in features:
        if feat in vects_lookup.keys():
            a_mat.append(vects[vects_lookup[feat]])
    a_mat_big = np.column_stack(a_mat)

    # predict
    dtest = xgb.DMatrix(a_mat_big)
    model = xgb.Booster({'nthread': 4})  # init model
    model.load_model(os.path.join(MODEL_DIR, '0001.model'))  # load model
    ypred = model.predict(dtest)

    # output
    id_ = 1
    with open(OUT_FILE, 'w') as out_:
        out_.write('id,cost\n')
        for pred in ypred:
            cost = math.pow(10, pred) - 1
            # transform predictions back to y with (10 ** pred) - 1
            out_.write('{},{}\n'.format(id_, cost))
            id_ += 1

if __name__ == '__main__':
    # extract_train('out_train.csv')
    # print('written out_train.csv')
    # extract_tube('out_tube.csv')
    # print('written out_tube.csv')
    # merge_train_tube('out_train.csv', 'out_tube.csv', 'out_train_merged.csv')
    # print('written out_train_merged.csv')
    # merge_test_tube()
    # print('written out_test.csv')
    print('preprocessing...')
    preprocess()
    FEATS = ['tube_assembly_id', 'diameter', 'wall', 'length', 'num_bends',
             'bend_radius', 'e0d_a_1x', 'e0d_a_2x', 'e0d_x_1x', 'e0d_x_2x',
             'a_form', 'x_form', 'adaptor', 'nut', 'sleeve', 'threaded',
             'boss', 'straight', 'elbow', 'other', 'float', 'hfl', 'tee',
             'total weight', 'S-0066', 'S-0041',
             'S-0072', 'S-0054', 'S-0026', 'S-0013', 'others', 'annual_usage',
             'min_order_quantity', 'bracket_pricing', 'quantity']
    print('training...')
    train(features=FEATS)
    print('predicting...')
    predict(features=FEATS)
    print('done')
