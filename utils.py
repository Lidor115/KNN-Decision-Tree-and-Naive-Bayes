def parser(path):
    data = []
    with open(path) as file:
        rows = file.readlines()
        for row in rows:
            data.append([feature.rstrip() for feature in row.split(sep='\t')])
    file.close()
    return data

def max_can_eat(examples):
    t = 0
    f = 0
    for example in examples:
        if example[1]:
            t += 1
        else:
            f += 1
    return t >= f, (t / (t + f))
def make_examples(train_p):
    attributes = {}
    train_p_T = [[train_p[j][i] for j in range(len(train_p))] for i in range(len(train_p[0]))]
    train_p_T.pop(-1)
    for row in train_p_T:
        attributes[row[0]] = set(row[1:])
    train_p.pop(0)
    index = 0
    all_examples = []
    for ex in train_p:
        tag = ex[-1]
        label = False
        if tag == 'yes':
            label = True
        index += 1
        all_examples.append((ex[:-1], label, index))
    return all_examples, attributes


def parseAttributes(first_line):
    index = 0
    all_attr = {}
    for a in first_line[:-1]:
        all_attr[a] = index
        index += 1
    return all_attr




def stringMaker_and_label(data):
    first_line = data.pop(0)
    string_label_data = []
    attributes = parseAttributes(first_line)
    index = 0
    for row in data:
        label = False
        tag = row[-1]
        if tag == 'yes':
            label = True
        s = ''.join([str(x) for x in row[:-1]])
        string_label_data.append((s, label, index))
        index += 1
    return string_label_data, attributes


def make_train_dev(path):
    num_lines = sum(1 for line in open(path))
    num_dev = int(num_lines * 0.2)
    dev_indexes = []
    for i in range(num_dev):
        x = randint(0, num_lines - 1)
        if not x in dev_indexes:
            dev_indexes.append(x)
    with open(path) as file:
        rows = file.readlines()
        with open('dev.txt', 'w+') as dev_file:
            with open('train.txt', 'w+') as train_file:
                dev_file.write(rows[0])
                train_file.write(rows[0])
                rows.pop(0)
                for i in range(1, num_lines - 1):
                    if i in dev_indexes:
                        dev_file.write(rows[i])
                    else:
                        train_file.write(rows[i])
            train_file.close()
        dev_file.close()
    file.close()


def hamming(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length.")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))


def dev_train_sep(k, data):
    div_k = []
    all_data_number = (len(data))
    fold = []
    total = 0
    one_fold = int(all_data_number / k)
    for i in range(k - 1):
        fold.append(one_fold)
        total += one_fold
    fold.append(int(all_data_number - total))
    for i in range(k):
        f = data[:fold[i]]
        data = data[fold[i]:]
        div_k.append(f)
    return div_k