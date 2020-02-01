""" A python script to split Equisetum into training, validation, and test set. """

import json
import os
from config import cfg
import random
import numpy as np


def read_data_from_file(json_filename):
    with open(json_filename) as f:
        species = json_filename.split('/')[-1].split('.')[0][:-1]
        file = json.load(f)
        img_anno_list = file['_via_img_metadata'].values()
        data = []
        # annotation for each image
        for img_anno in img_anno_list:
            img_file_name = os.path.join(species, img_anno['filename'])  # e.g. ferrissii/175085.jpeg
            anno_info = []
            region_list = img_anno['regions']
            for region in region_list:
                bbox = region['shape_attributes']
                if bbox['name'] == 'rect':
                    type = region["region_attributes"]["type"]
                    if cfg['is_useful_anno'][type] == 0:
                        continue  # omit unnecessary labeled type
                    xs, ys = bbox['x'], bbox['y']
                    h, w = bbox['height'], bbox['width']
                    HL_label = cfg['class'][species] # mapping name to class number
                    record = [xs, ys, xs + w, ys + h, HL_label]  # [xmin, ymin, xmax, ymax, label]
                    anno_info.append(record)
            if anno_info == []:
                continue
            img_info = {
                'imageFileName': img_file_name,
                'species': species,
                'annotation': anno_info
            }
            data.append(img_info)
        return data


def split_dataset(training_percent, test_percent):
    train_data = []
    val_data = []
    test_data = []
    for json_name in os.listdir(cfg['anno_dir']):
        if json_name.split('.')[-1] != 'json':
            continue
        json_path = os.path.join(cfg['anno_dir'], json_name)
        annotations = read_data_from_file(json_path)
        random.shuffle(annotations)
        total_num = len(annotations)
        train_num = int(np.floor(training_percent * total_num))
        test_num = int(np.floor(test_percent * total_num))
        train_data += annotations[:train_num]
        test_data += annotations[train_num:train_num + test_num]
        val_data += annotations[train_num + test_num:]
    return train_data, test_data, val_data


def get_json_dataset(training_percent, test_percent):
    train_data, test_data, val_data = split_dataset(training_percent, test_percent)
    with open(cfg['main_dir'] + '/Equisetum/training_set.json', 'w') as tr_set:
        json.dump(train_data, tr_set)
    with open(cfg['main_dir'] + '/Equisetum/val_set.json', 'w') as v_set:
        json.dump(val_data, v_set)
    with open(cfg['main_dir'] + '/Equisetum/test_set.json', 'w') as te_set:
        json.dump(test_data, te_set)


if __name__ == '__main__':
    # the percentage of each set (val = 1 - training_percent - test_percent)
    training_percent = 0.5
    test_percent = 0.3
    get_json_dataset(training_percent, test_percent)
