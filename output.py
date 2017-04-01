import os

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'competition_data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train_set.csv')
IN_FILE = os.path.join(DATA_DIR, 'test_set.csv')
OUT_FILE = os.path.join(CUR_DIR, 'out.csv')
EXTRACT_FILE = os.path.join(CUR_DIR, 'extracted.csv')

# put random price
# with open(OUT_FILE, 'w') as out_:
#     out_.write('id,cost\n')
#     for id_ in IDS:
#         out_.write('{},1\n'.format(id_))


def extract_data():
    with open(TRAIN_FILE, 'r') as in_, open(EXTRACT_FILE, 'w') as out_:
        tmp = in_.readlines()
        head = tmp[0]
        out_.write(head)
        for line in tmp[1:]:
            values = line.strip().split(',')
            if int(values[3]) == 0 and int(values[4]) == 0 and values[5] == 'Yes' and int(values[6]) == 1:
                out_.write(line)


if __name__ == '__main__':
    extract_data()
