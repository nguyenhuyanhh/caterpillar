"""
Caterpillar Tube Pricing

Nguyen Huy Anh, Lee Vicson, Deon Seng, Oh Yoke Chew
"""

import math
import os

import numpy as np
import xgboost as xgb

# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'competition_data')
# inputs
TRAIN_FILE = os.path.join(DATA_DIR, 'train_set.csv')
TUBE_FILE = os.path.join(DATA_DIR, 'tube_mod.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_set.csv')
# constants
SUPP_ENCODE = ['S-0066', 'S-0041', 'S-0072',
               'S-0054', 'S-0026', 'S-0013', 'others']
# outputs
OUT_FILE = os.path.join(CUR_DIR, 'out_latest.csv')


def extract_train(out_file):
    """
    Extract train.csv, with one-hot encoding for supplier

    Arguments:
        out_file: str - path to output file
    CSV header:
        tube_assembly_id,S-0066,S-0041,S-0072,S-0054,S-0026,S-0013,others,
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
            encoding = ['0', '0', '0', '0', '0', '0', '0']
            if values[1] in SUPP_ENCODE:
                index = SUPP_ENCODE.index(values[1])
                encoding[index] = '1'
            else:
                encoding[-1] = '1'
            value_tmp = [values[0]] + encoding + values[-5:]
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
    with open('tube_total_weight.csv','r') as weight_:
        total_weight = weight_.readlines()
    with open(out_file, 'w') as out_:
        head = tmp[0].strip().split(',')
        head2 = total_weight[0].strip().split(',')
        head3 = ['weighted','not weighted']
        head_tmp = [head[0]] + head[2:13] + head2[1:-1] + head3 + [head2[-1]]
        out_.write(','.join(head_tmp) + '\n')
        i = 1
        for line in tmp[1:]:
            weight_raw = total_weight[i].strip().split(',')
            weight = [weight_raw[-1]]
            encoding = ['0','0']
            if(weight[0] == '0'):
                encoding[1] = '1'
            else:
                encoding[0] = '1'
            values = line.strip().split(',')
            values_tmp = [values[0]] + values[2:13] + weight_raw[1:-1] + encoding + weight
            out_.write(','.join(values_tmp) + '\n')
            i=i+1


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


def train():
    """
    Build the model for prediction.
    """
    # get coefficients
    y_vect = list()
    id_vect = list()
    diameter_vect = list()
    wall_vect = list()
    length_vect = list()
    num_bends_vect = list()
    bend_radius_vect = list()
    s66 = list()
    s41 = list()
    s72 = list()
    s54 = list()
    s26 = list()
    s13 = list()
    others = list()
    min_q = list()
    usage = list()
    weighted = list()
    not_weighted = list()
    total_weight = list()
    adaptor = list()
    nut = list()
    sleeve = list()
    threaded = list()
    boss = list()
    straight = list()
    elbow = list()
    other = list()
    float_ = list()
    hfl = list()
    tee = list()
    quantity_vect = list()
    bracket_vect = list()
    t1 = list()
    t2 = list()
    t3 = list()
    t4 = list()
    t5 = list()
    t6 = list()
    with open(os.path.join(CUR_DIR, 'out_train_merged.csv'), 'r') as merged_:
        tmp = merged_.readlines()[1:]
        for line in tmp:
            values = line.strip().split(',')

            transform = math.log10(float(values[-1]) + 1)
            # transform y to log10(1+y)
            y_vect.append(transform)

            id_vect.append(float(values[0][-5:]))
            diameter_vect.append(float(values[1]))
            wall_vect.append(float(values[2]))
            length_vect.append(float(values[3]))
            num_bends_vect.append(float(values[4]))
            bend_radius_vect.append(float(values[5]))
            weighted.append(int(values[-15]))
            not_weighted.append(int(values[-14]))
            transform = math.log10(float(values[-13]) + 1)
            # transform y to log10(1+y)
            total_weight.append(transform)
            t1.append(int(values[-32]))
            t2.append(int(values[-31]))
            t3.append(int(values[-30]))
            t4.append(int(values[-29]))
            t5.append(int(values[-28]))
            t6.append(int(values[-27]))
            adaptor.append(int(values[-26]))
            nut.append(int(values[-25]))
            sleeve.append(int(values[-24]))
            threaded.append(int(values[-23]))
            boss.append(int(values[-22]))
            straight.append(int(values[-21]))
            elbow.append(int(values[-20]))
            other.append(int(values[-19]))
            float_.append(int(values[-18]))
            hfl.append(int(values[-17]))
            tee.append(int(values[-16]))
            s66.append(int(values[-12]))
            s41.append(int(values[-11]))
            s72.append(int(values[-10]))
            s54.append(int(values[-9]))
            s26.append(int(values[-8]))
            s13.append(int(values[-7]))
            others.append(int(values[-6]))
            usage.append(int(values[-5]))
            min_q.append(int(values[-4]))
            quantity_vect.append(float(values[-2]))
            if values[-3] == 'Yes':
                bracket_vect.append(1)
            else:
                bracket_vect.append(0)
    a_mat = [id_vect,diameter_vect, wall_vect, length_vect, num_bends_vect, bend_radius_vect, t1,t2,t3,t4,t5,t6,adaptor, nut, sleeve, threaded, boss, straight, elbow, other, float_, hfl, tee,total_weight,
             s66, s41, s72, s54, s26, s13, others, usage, min_q, quantity_vect, bracket_vect]
    a_mat_big = np.column_stack(a_mat)

    dtrain = xgb.DMatrix(a_mat_big, label=y_vect)
    param = {'max_depth': 8,
             'eta': 0.3,
             'min_child_weight': 5}
    # I have no clue what any of these do, but yeah
    num_round = 30
    """
    # using the built in cv method to check errors, it uses rmse though
    xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'rmse'}, seed = 0,
       callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
    """
    model = xgb.train(param, dtrain, num_round)
    # This is the model
    model.save_model(os.path.join(CUR_DIR, '0001.model'))


def predict():
    """
    Predict based on the model.
    """
    id_vect = list()
    diameter_vect = list()
    wall_vect = list()
    length_vect = list()
    num_bends_vect = list()
    bend_radius_vect = list()
    s66 = list()
    s41 = list()
    s72 = list()
    s54 = list()
    s26 = list()
    s13 = list()
    others = list()
    weighted = list()
    not_weighted = list()
    total_weight = list()
    adaptor = list()
    nut = list()
    sleeve = list()
    threaded = list()
    boss = list()
    straight = list()
    elbow = list()
    other = list()
    float_ = list()
    hfl = list()
    tee = list()
    min_q = list()
    usage = list()
    quantity_vect = list()
    bracket_vect = list()
    t1 = list()
    t2 = list()
    t3 = list()
    t4 = list()
    t5 = list()
    t6 = list()
    with open(os.path.join(CUR_DIR, 'out_test.csv'), 'r') as merged_:
        tmp = merged_.readlines()
        for line in tmp:
            values = line.strip().split(',')

            id_vect.append(float(values[0][-5:]))
            diameter_vect.append(float(values[1]))
            wall_vect.append(float(values[2]))
            length_vect.append(float(values[3]))
            num_bends_vect.append(float(values[4]))
            bend_radius_vect.append(float(values[5]))
            weighted.append(int(values[-14]))
            not_weighted.append(int(values[-13]))
            transform = math.log10(float(values[-12]) + 1)
            # transform y to log10(1+y)
            total_weight.append(transform)
            t1.append(int(values[-31]))
            t2.append(int(values[-30]))
            t3.append(int(values[-29]))
            t4.append(int(values[-28]))
            t5.append(int(values[-27]))
            t6.append(int(values[-26]))
            adaptor.append(int(values[-25]))
            nut.append(int(values[-24]))
            sleeve.append(int(values[-23]))
            threaded.append(int(values[-22]))
            boss.append(int(values[-21]))
            straight.append(int(values[-20]))
            elbow.append(int(values[-19]))
            other.append(int(values[-18]))
            float_.append(int(values[-17]))
            hfl.append(int(values[-16]))
            tee.append(int(values[-15]))
            s66.append(int(values[-11]))
            s41.append(int(values[-10]))
            s72.append(int(values[-9]))
            s54.append(int(values[-8]))
            s26.append(int(values[-7]))
            s13.append(int(values[-6]))
            others.append(int(values[-5]))
            usage.append(int(values[-4]))
            min_q.append(int(values[-3]))
            quantity_vect.append(float(values[-1]))
            if values[-2] == 'Yes':
                bracket_vect.append(1)
            else:
                bracket_vect.append(0)
    a_mat = [id_vect,diameter_vect, wall_vect, length_vect, num_bends_vect, bend_radius_vect, t1,t2,t3,t4,t5,t6,adaptor, nut, sleeve, threaded, boss, straight, elbow, other, float_, hfl, tee,total_weight,
             s66, s41, s72, s54, s26, s13, others, usage, min_q, quantity_vect, bracket_vect]
    a_mat_big = np.column_stack(a_mat)

    dtest = xgb.DMatrix(a_mat_big)
    model = xgb.Booster({'nthread': 4})  # init model
    model.load_model(os.path.join(CUR_DIR, '0001.model'))  # load model
    ypred = model.predict(dtest)
    # predict based on model

    id_ = 1
    with open(OUT_FILE, 'w') as out_:
        out_.write('id,cost\n')
        for pred in ypred:
            cost = math.pow(10, pred) - 1
            # transform predictions back to y with (10 ** pred) - 1
            out_.write('{},{}\n'.format(id_, cost))
            id_ += 1

if __name__ == '__main__':
    extract_train('out_train.csv')
    print('written out_train.csv')
    extract_tube('out_tube.csv')
    print('written out_tube.csv')
    merge_train_tube('out_train.csv', 'out_tube.csv', 'out_train_merged.csv')
    print('written out_train_merged.csv')
    merge_test_tube()
    print('written out_test.csv')
    print('training...')
    train()
    print('predicting...')
    predict()
    print('done')
