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

# put random price
# with open(OUT_FILE, 'w') as out_:
#     out_.write('id,cost\n')
#     for id_ in IDS:
#         out_.write('{},1\n'.format(id_))


def extract_train():
    with open(TRAIN_FILE, 'r') as in_, open(EXTRACT_TRAIN_FILE, 'w') as out_:
        tmp = in_.readlines()
        head = tmp[0].strip().split(',')
        out_.write(head[0] + ',' + head[7] + '\n')
        for line in tmp[1:]:
            values = line.strip().split(',')
            if int(values[3]) == 0 and int(values[4]) == 0 and values[5] == 'Yes' and int(values[6]) == 1:
                out_.write(values[0] + ',' + values[7] + '\n')


def extract_tubes():
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


if __name__ == '__main__':
    extract_train()
    extract_tubes()
