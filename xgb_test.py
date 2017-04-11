"""
Caterpillar Tube Pricing

Nguyen Huy Anh, Lee Vicson, Deon Seng, Oh Yoke Chew
"""

import os
import math

import numpy as np
import xgboost as xgb

# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'competition_data')
# inputs
TRAIN_FILE = os.path.join(DATA_DIR, 'train_set.csv')
TUBE_FILE = os.path.join(DATA_DIR, 'tube.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_set.csv')
#CONSTANTS
SUPP_ENCODE = ['S-0066','S-0041','S-0072','S-0054','S-0026','S-0013','others']
# outputs
OUT_FILE = os.path.join(CUR_DIR, 'out.csv')

def extract_train():
    """
    Modified to extract all train data
    CSV header: tube_assembly_id,bracket,quantity,cost
    """
    tmp = list()
    with open(TRAIN_FILE, 'r') as in_:
        tmp = in_.readlines()
    with open('out_train.csv', 'w') as out_:
        head = tmp[0].strip().split(',')
        head_tmp = [head[0]] + SUPP_ENCODE + head[-2:]
        out_.write(','.join(head_tmp) + '\n')
        for line in tmp[1:]:
            values = line.strip().split(',')
            encoding = ['0','0','0','0','0','0','0']
            if values[1] in SUPP_ENCODE:
                index = SUPP_ENCODE.index(values[1])
                encoding[index] = '1'
            else:
                encoding[-1] = '1'
            value_tmp = [values[0]] + encoding + values[-3:]
            out_.write(','.join(value_tmp) + '\n')

def extract_tube(out_file):
    """
    Extract tube.csv.

    Arguments:
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
            values_tmp = [values[0]] + values[2:7]
            out_.write(','.join(values_tmp) + '\n')

def merge_train_tube(in_train_file, in_tube_file, out_file):
    """
    Merge two data sets from extract_train and extract_tube together.

    Arguments:
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
                continue

def merge_test_tube():
    with open(TEST_FILE, 'r') as in_:
        tmp_test = in_.readlines()[1:]
    with open(TUBE_FILE, 'r') as tube_:
        tmp_tube = tube_.readlines()
    with open(os.path.join(CUR_DIR,'out_test.csv'), 'w') as out_:
        for line in tmp_test:
            tmp = line.strip().split(',')
            tube_id = int(tmp[1][-5:])
            if tube_id > 19490:
                tube_id = tube_id - 1
            tube_info = ((tmp_tube[tube_id]).strip().split(','))[2:7]
            out_tmp = [tmp[1]] + tube_info + tmp[-2:]
            out_.write(','.join(out_tmp) + '\n')
        

def train():
    # get coefficients
    y_vect = list()
    id_vect = list()
    diameter_vect = list()
    wall_vect = list()
    length_vect = list()
    num_bends_vect = list()
    bend_radius_vect = list()
    quantity_vect = list()
    bracket_vect = list()
    with open(os.path.join(CUR_DIR,'out_merged.csv'), 'r') as merged_:
        tmp = merged_.readlines()[1:]
        for line in tmp:
            values = line.strip().split(',')

            transform = math.log10(float(values[-1])+1)
            # transform y to log10(1+y)
            y_vect.append(transform)

            id_vect.append(float(values[0][-5:]))
            diameter_vect.append(float(values[1]))
            wall_vect.append(float(values[2]))
            length_vect.append(float(values[3]))
            num_bends_vect.append(float(values[4]))
            bend_radius_vect.append(float(values[5]))
            quantity_vect.append(float(values[-2]))
            if values[-3] == 'Yes':
                bracket_vect.append(1)
            else:
                bracket_vect.append(0)
    a_mat = [id_vect, diameter_vect, wall_vect, length_vect,
             num_bends_vect, bend_radius_vect, quantity_vect, bracket_vect]
    a_mat_big = np.column_stack(a_mat)

    dtrain = xgb.DMatrix(a_mat_big, label=y_vect)
    param = {'max_depth':8,
             'eta':0.3,
             'min_child_weight':5}
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
    model.save_model(os.path.join(CUR_DIR,'0001.model'))

def predict():
    id_vect = list()
    diameter_vect = list()
    wall_vect = list()
    length_vect = list()
    num_bends_vect = list()
    bend_radius_vect = list()
    quantity_vect = list()
    bracket_vect = list()
    with open(os.path.join(CUR_DIR,'out_test.csv'), 'r') as merged_:
        tmp = merged_.readlines()
        for line in tmp:
            values = line.strip().split(',')
            
            id_vect.append(float(values[0][-5:]))
            diameter_vect.append(float(values[1]))
            wall_vect.append(float(values[2]))
            length_vect.append(float(values[3]))
            num_bends_vect.append(float(values[4]))
            bend_radius_vect.append(float(values[5]))
            quantity_vect.append(float(values[-1]))
            if values[-2] == 'Yes':
                bracket_vect.append(1)
            else:
                bracket_vect.append(0)
    a_mat = [id_vect, diameter_vect, wall_vect, length_vect,
             num_bends_vect, bend_radius_vect, quantity_vect, bracket_vect]
    a_mat_big = np.column_stack(a_mat)

    dtest = xgb.DMatrix(a_mat_big)
    model = xgb.Booster({'nthread':4}) #init model
    model.load_model(os.path.join(CUR_DIR,'0001.model')) # load model
    ypred = model.predict(dtest)
    # predict based on model
    
    id = 1
    with open(OUT_FILE, 'w') as out_:
        out_.write('id,cost\n')
        for pred in ypred:
            cost = math.pow(10,pred) - 1
            # transform predictions back to y with (10 ** pred) - 1
            out_.write('{},{}\n'.format(id, cost))
            id += 1

if __name__ == '__main__':
    extract_train()
    """try:
        train()
        predict()
        print('done')
        input()
    except Exception as e:
        print(e.args)
        print("Press Enter to continue ..." )
        input() """

