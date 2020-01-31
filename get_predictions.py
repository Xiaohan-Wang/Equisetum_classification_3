import os
import torch
from utils.ssd import build_ssd
import cv2
import numpy as np
from utils.transform import BaseTransform
import json
from tqdm import tqdm
import warnings
from config import cfg


warnings.filterwarnings('ignore')


def nms(dets, thresh):
    """Pure Python NMS baseline."""
    dets = np.array(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  # score of bbox

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # sort score in descending order
    order = scores.argsort()[::-1]
    keep = []  # keep final bbox
    while order.size > 0:
        # order[0] is kept since it has the highest score
        i = order[0]
        keep.append(dets[i].tolist())
        # computing overlapping area of windows i and all the other window
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # compute iou = intersection area / union area
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # indexes of windows whose iou with window i are less than threshold, other windows are absorbed by window i
        inds = np.where(ovr <= thresh)[0]
        # keep the windows whose IoU with windows i are less than threshold order (+ 1 is needed since the length if ovr is 1 less than order)
        order = order[inds + 1]

    return keep


def predict(frames, transform, net, tile_number, cuda):
    # tile_number: how many tiles has been computed
    height, width = frames.shape[1:3]
    x = []
    for i in range(frames.shape[0]):
        x.append(transform(frames[i])[0])
    x = np.stack(x, axis=0)
    x = torch.from_numpy(x).permute(0, 3, 1, 2)
    if cuda:
        x = x.cuda()
        net = net.cuda()
    with torch.no_grad():
        y = net(x)  # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])
    bbox = []
    for k in range(detections.size(0)):
        for i in range(detections.size(1)):
            j = 0
            while detections[k, i, j, 0] > 0:
                pt = (detections[k, i, j, 1:] * scale).cpu().numpy()
                bbox.append({
                    'score': float(detections[k, i, j, 0].cpu().numpy()),
                    'tile': k + tile_number,  # store the tile index to compute offset of bbox
                    'index': i,  # class index
                    'bbox': pt  # [xs, ys, se, ye]
                })
                j += 1
    return bbox


def test_img(cuda, net_filepath, img_file, tile, overlap, batch_size, iou_thresh):
    # load net
    net = build_ssd('test', 300, cfg['num_classes'])  # initialize SSD
    if cuda:
        net.load_state_dict(torch.load(net_filepath))
    else:
        net.load_state_dict(torch.load(net_filepath, map_location=torch.device('cpu')))
    net.eval()
    print('Finished loading model!')
    # load data
    transform = BaseTransform(net.size, np.array(cfg['mean']))
    with open(img_file, 'r') as f:
        labels = json.load(f)

    data = {}
    for i in tqdm(range(len(labels))):
        item = labels[i]
        img = cv2.imread(os.path.join(cfg['img_dir'], item['imageFileName']))
        img = img[:, :, (2, 1, 0)].copy()  # to rgb
        h, w, c = img.shape
        imgs = []
        stride = tile - overlap
        h_num = (h - tile) // stride + 1
        w_num = (w - tile) // stride + 1
        for i in range(h_num):
            for j in range(w_num):
                # split the image into tiles so that the test image size is the same as training data
                x = img[i * stride:(i * stride + tile), j * stride:(j * stride + tile), :]
                imgs.append(x)
        # stack tiles
        input = np.stack(imgs, axis=0)
        bbox = []
        for i in range((input.shape[0] - 1) // batch_size + 1):
            bbox += predict(input[batch_size * i:batch_size * (i + 1)], transform, net, i * batch_size, cuda)

        dets = []
        for i in range(len(bbox)):
            xs, ys, xe, ye = bbox[i]['bbox'][:]
            tile_ind = bbox[i]['tile']
            class_index = bbox[i]['index']
            xdiff = xe - xs
            ydiff = ye - ys
            row_num = tile_ind // w_num
            col_num = tile_ind % w_num
            # compute offset
            ys += row_num * stride
            xs += col_num * stride
            xe = xs + xdiff
            ye = ys + ydiff
            score = bbox[i]['score']
            dets.append([xs, ys, xe, ye, score, class_index])

        keep = nms(dets, iou_thresh)
        data[item['imageFileName']] = keep
    return data


def get_predictions(cuda, net_filepath, tile, overlap, batch_size, iouthresh):
    if not os.path.exists(os.path.join(cfg['main_dir'], 'results')):
        os.mkdir(os.path.join(cfg['main_dir'], 'results'))
    if not os.path.exists(os.path.join(cfg['main_dir'], 'results', 'predictions')):
        os.mkdir(os.path.join(cfg['main_dir'], 'results', 'predictions'))

    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # get prediction for training, validation and test set images
    for file in ['training_set', 'val_set', 'test_set']:
        test_img_file = cfg[file]
        data = test_img(cuda, net_filepath, test_img_file, tile, overlap, batch_size, iouthresh)
        with open(os.path.join(cfg['main_dir'], 'results', 'predictions', file + '.json'), 'w') as f:
            json.dump(data, f)




if __name__ == '__main__':
    # use the pretrained net or your own model
    net_filepath = cfg['save_folder'] + cfg['name'] + '/model.pth'
    print(net_filepath)

    tile = 500  # split the image into tiles so that the test image size is similar to training image size
    overlap = 100  # overlap between two tiles
    batch_size = 8  # how many tiles are processed by one inference
    iouthresh = 0.1 # supress overlapped prediction window

    cuda = True

    # get prediction for training, validation and test set images
    get_predictions(cuda, net_filepath, tile, overlap, batch_size, iouthresh)
