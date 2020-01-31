import torch
import torch.utils.data as data
import cv2
import numpy as np
from utils.augmentations import SSDAugmentation
import json
import os
from config import cfg


class EquisetumDataset(data.Dataset):
    """Equisetum dataset"""

    def __init__(self, filepath, transform=None):
        self.transform = transform
        self.data = []
        with open(filepath, 'r') as f:
            data = json.load(f)
        # load annotations for each image except for ferrissii
        for i in range(len(data)):
            # We assign all nodes in hyemale as H node, and all nodes in laevigatum as L node. However, since ferrissii
            # contains nodes of both types, we cannot use it as our training data for node detection task since we are
            # not sure the node type (H node or L node) for nodes in ferrissii.
            if data[i]['species'] != 'ferrissii':
                self.data.append(data[i])

    def __getitem__(self, index):
        img, target, h, w = self.pull_item(index)
        return torch.from_numpy(img).permute(2, 0, 1), target

    def __len__(self):
        return len(self.data)

    def pull_item(self, index):
        item = self.data[index]
        img = cv2.imread(os.path.join(cfg['img_dir'], item['imageFileName']))
        height, width, channels = img.shape
        img = img[:, :, (2, 1, 0)]  # to rgb
        target = np.array(item['annotation']).astype(float)
        # to macth the transform format (xs, ys xe, ye are between 0 and 1)
        target[:, 0], target[:, 2] = target[:, 0] / width, target[:, 2] / width
        target[:, 1], target[:, 3] = target[:, 1] / height, target[:, 3] / height

        if self.transform is not None:
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            height, width, channels = img.shape

        return img, target, height, width

#     def draw_boxes(self, index):
#         img, target, h, w = self.pull_item(index)
#         img = img.copy()
#         for box in target:
#             xs, ys, xe, ye = box[:4]
#             label = box[4]
#             img_box = cv2.rectangle(img, (int(xs * w), int(ys * h)), (int(xe * w), int(ye * h)), (0, 0, 255), 3)
#         return img_box
#
#
# if __name__ == '__main__':
#     training_set = '/Users/xiaohan/research/Equisetum_new/Equisetum/training_set.json'
#     ds = EquisetumDataset(training_set, SSDAugmentation())
#     for i in range(10):
#         img = ds.draw_boxes(i)
#         name = 'example' + str(i) + '.jpg'
#         cv2.imwrite(name, img)