import os

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'competition_data')
IN_FILE = os.path.join(DATA_DIR, 'test_set.csv')
OUT_FILE = os.path.join(CUR_DIR, 'out.csv')

# get ids
with open(IN_FILE, 'r') as in_:
    IDS = [line.strip().split(',')[0] for line in in_.readlines()][1:]

# put random price
with open(OUT_FILE, 'w') as out_:
    out_.write('id,cost\n')
    for id_ in IDS:
        out_.write('{},1\n'.format(id_))
