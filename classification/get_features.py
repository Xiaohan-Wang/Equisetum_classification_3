import json
import csv
from ..config import cfg
import numpy as np


def score_as_key(elem):
    return elem[-2]


def statistics(boxes, k=10):
    h_node = []
    l_node = []
    for box in boxes:
        if cfg['node_class'][box[-1]] == 'HNode':
            h_node.append(box)
        elif cfg['node_class'][box[-1]] == 'LNode':
            l_node.append(box)
    h_node.sort(key=score_as_key, reverse=True)
    l_node.sort(key=score_as_key, reverse=True)
    H_score = []
    L_score = []
    for i in range(k):
        H_score.append(h_node[i][-2])
        L_score.append(l_node[i][-2])
    H_score = np.array(H_score)
    L_score = np.array(L_score)
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
    return info, h_node[:k], l_node[:k]
#
# def statistics(boxes, k, thresh):
#     h_node = []
#     l_node = []
#     for box in boxes:
#         if cfg['HL_map_anno'][box[-1]] == 'HNode':
#             if box[-2] >= thresh[0]:
#                 h_node.append(box)
#         elif cfg['HL_map_anno'][box[-1]] == 'LNode':
#             if box[-2] >= thresh[1]:
#                 l_node.append(box)
#     H_score = []
#     L_score = []
#     for i in range(len(h_node)):
#         H_score.append(h_node[i][-2])
#     for i in range(len(l_node)):
#         L_score.append(l_node[i][-2])
#     H_num = len(h_node)
#     L_num = len(l_node)
#     # H_score = np.array(H_score)
#     # L_score = np.array(L_score)
#     # H_mean = round(np.mean(H_score), 5)
#     # H_std = round(np.std(H_score), 5)
#     # L_mean = round(np.mean(L_score), 5)
#     # L_std = round(np.std(L_score), 5)
#     # mean_HL = round(H_mean / L_mean, 5)
#     # std_HL = round(H_std / L_std, 5)
#     # info = [
#     #     H_num/(H_num+L_num) if H_num+L_num != 0 else 0,
#     #     # H_mean,
#     #     L_num/(H_num+L_num) if H_num+L_num != 0 else 0,
#     #     # L_mean,
#     # ]
#     info = [
#         H_num,
#         # H_mean,
#         L_num,
#         # L_mean,
#     ]
#     return info, h_node[:k], l_node[:k]

def extract_data(filepath, k, save_filepath):
    # prepare csv file for statistics
    with open(save_filepath, 'w') as f:
        csv_write = csv.writer(f)
        header = ["img_name",
                  "H_mean",
                  "H_std",
                  "L_mean",
                  "L_std",
                  "mean_H/L",
                  "std_H/L"]
        csv_write.writerow(header)

    with open(filepath) as f:
        data = json.load(f)

    for img in data.keys():
        keep = data[img]
        info, h_top, l_top = statistics(keep, k)
        boxes_info = [img] + info
        with open(save_filepath, "a+") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(boxes_info)

# def extract_data(filepath, k, save_filepath, thresh):
#     # prepare csv file for statistics
#     with open(save_filepath, 'w') as f:
#         csv_write = csv.writer(f)
#         header = ["img_name",
#                   'H_percentage'
#                   # "H_mean",
#                   # "H_std",
#                   'L_percentage']
#                   # "L_mean",
#                   # "L_std",
#                   # 'H/L'
#                   # "mean_H/L",
#                   # "std_H/L"]
#         csv_write.writerow(header)
#
#     with open(filepath) as f:
#         data = json.load(f)
#
#     for img in data.keys():
#         keep = data[img]
#         info, h_top, l_top = statistics(keep, k, thresh)
#         boxes_info = [img] + info
#         with open(save_filepath, "a+") as f:
#             csv_write = csv.writer(f)
#             csv_write.writerow(boxes_info)


if __name__ == '__main__':
    filepath = '/Users/xiaohan/research/Equisetum/code/result/predictions/val_predictions_iou_0.1.json'
    k = 10
    # save_filepath = '/Users/xiaohan/research/Equisetum/code/result/features/test_top' + str(k) + '.csv'
    # extract_data(filepath, k, save_filepath)

    save_filepath = '/Users/xiaohan/research/Equisetum/code/result/features/val_HL_num.csv'
    thresh = [0.6, 0.7]
    extract_data(filepath, k, save_filepath, thresh)
