import json
import csv
import numpy as np
import os
from ..config import cfg

def score_as_key(elem):
    return elem[-2]

def get_top_bbox(bboxs, k=10):
    bboxs = np.array(bboxs)
    h_node = bboxs[bboxs[:, -1] == 1, :]
    l_node = bboxs[bboxs[:, -1] == 2, :]
    # get the highest k scores of hnodes and lnodes seperately
    h_node = h_node[h_node[:, -2].argsort()[::-1][:k]]
    l_node = l_node[l_node[:, -2].argsort()[::-1][:k]]
    H_score = h_node[:, -2]
    L_score = l_node[:, -2]

    H_mean = round(np.mean(H_score), 5)
    H_std = round(np.std(H_score), 5)
    L_mean = round(np.mean(L_score), 5)
    L_std = round(np.std(L_score), 5)
    mean_HL = round(H_mean / L_mean, 5)
    std_HL = round(H_std / L_std, 5)
    info = [
        H_mean,
        H_std,
        L_mean,
        L_std,
        mean_HL,
        std_HL
    ]
    return info, h_node, l_node

def save_top_k_info(set_name, k, save_dir):
    predictions_path = os.path.join(cfg['main_dir']+'/results/predictions', set_name + '.json')
    with open(predictions_path) as f:
        predictions = json.load(f)

    # prepare csv file for statistics
    save_path = os.path.join(save_dir, set_name+'.csv')
    with open(save_path, 'w') as f:
        csv_write = csv.writer(f)
        header = ["img_name", "H_mean", "H_std", "L_mean", "L_std", "mean_H/L", "std_H/L"]
        csv_write.writerow(header)

    for img in predictions.keys():
        keep = predictions[img]
        info, h_top, l_top = get_top_bbox(keep, k)
        boxes_info = [img] + info
        with open(save_path, "a+") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(boxes_info)


if __name__ == '__main__':
    save_dir = cfg['main_dir'] + '/results/features'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for set_name in ['training_set', 'test_set']:
        save_top_k_info(set_name, 10, save_dir)
