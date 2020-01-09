from random import randint
from sys import argv


def parser(path):
    data = []
    with open(path) as file:
        rows = file.readlines()
        for row in rows:
            data.append([feature.rstrip() for feature in row.split(sep='\t')])
    file.close()
    return data

def stringMaker_and_label(data):
    first_line = data.pop(0)
    string_label_data = []
    for row in data:
        label = False
        tag = row[0]
        if tag == 'yes':
            label = True
        s =''.join([str(x) for x in row[:-1]])
        string_label_data.append((s,label))
    return string_label_data

def make_train_dev(path):
    num_lines = sum(1 for line in open(path))
    num_dev = int(num_lines*0.2)
    dev_indexes =[]
    for i in range(num_dev):
        x =randint(0,num_lines-1)
        if not x in dev_indexes:
            dev_indexes.append(x)
    with open(path) as file:
        rows = file.readlines()
        with open('dev.txt','w+') as dev_file:
            with open('train.txt', 'w+') as train_file:
                dev_file.write(rows[0])
                train_file.write(rows[0])
                rows.pop(0)
                for i in range(1,num_lines-1):
                    if i in dev_indexes:
                        dev_file.write(rows[i])
                    else:
                        train_file.write(rows[i])
            train_file.close()
        dev_file.close()
    file.close()

def hamming(s1, s2):
    return


if __name__ == '__main__':
    path = parser(argv[1])
    make_train_dev(argv[1])
    s = stringMaker_and_label(path)
    num_lines = sum(1 for line in open(argv[1]))
    print(int(num_lines *0.2))
    print(s[5][0])