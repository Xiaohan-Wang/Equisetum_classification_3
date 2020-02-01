from sklearn.linear_model import LogisticRegression
from ..config import cfg
import numpy as np
import csv
import os


def extract_statistics(statistics_path):
    statistics = []
    with open(statistics_path, 'r') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            statistics.append(line)
    X = []
    Y = []
    imgs = []
    for item in statistics[1:]:
        name = item[0]
        label = cfg['class'][name.split('/')[0]]
        # feature = [float(entry) for entry in item[1:]]
        feature = list(map(float, item[1:]))
        X.append(feature)
        Y.append(label)
        imgs.append(name)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y, imgs


def classifier(statistics_path):
    X, Y, _ = extract_statistics(statistics_path)
    classifier = LogisticRegression()
    classifier.fit(X, Y)
    es = classifier.predict(X)
    acc = np.mean((es == Y))
    return classifier, acc


def predict(statistics_path, classifier):
    X, gt, imgs = extract_statistics(statistics_path)
    es = classifier.predict(X)
    acc = np.mean((gt == es))
    return imgs, es, gt, acc


def confusion_table(es, gt):
    save_dir = cfg['main_dir'] + '/results/classification'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    table = np.zeros([3, 3])
    for i in range(len(gt)):
        table[gt[i]][es[i]] += 1
    with open(save_dir + '/confusion_table.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([' ', 'hyemale', 'laevigatum', 'ferrissii'])
        writer.writerow(['hyemale'] + table[0].tolist())
        writer.writerow(['laevigatum'] + table[1].tolist())
        writer.writerow(['ferrissii'] + table[2].tolist())
    return table


if __name__ == '__main__':
    train_statistics_path = cfg['main_dir'] + '/results/features/training_set.csv'
    test_statistics_path = cfg['main_dir'] + '/results/features/test_set.csv'

    lr_classifier, train_acc = classifier(train_statistics_path)
    imgs, es, gt, test_acc = predict(test_statistics_path, lr_classifier)

    confusion_table(es, gt)

    print("train acc: ", train_acc)
    print("test acc: ", test_acc)
