import json
from ..config import cfg
import cv2
import os
import numpy as np


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=15):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            img = cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                img = cv2.line(img, s, e, color, thickness)
            i += 1
    return img


def drawpoly(img, pts, color, thickness=1, style='dotted', ):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        img = drawline(img, s, e, color, thickness, style)
    return img


def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    img = drawpoly(img, pts, color, thickness, style)
    return img


def draw_box(gt_path, pre_path, thresh, save_folder):
    with open(gt_path) as f:
        gt = json.load(f)
    with open(pre_path) as f:
        pre = json.load(f)

    thickness = 10
    color = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # [bg, class1, class2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(gt)):
        item_gt = gt[i]
        name = item_gt["imageFileName"]
        gt_anno = item_gt["annotation"]
        pre_result = pre[name]
        img = cv2.imread(os.path.join(cfg['img_dir'], name))
        # img = img[:, :, (2, 1, 0)].copy()  # to rgb
        for box in gt_anno:
            img = drawrect(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color[0], thickness, style='dash')
        for box in pre_result:
            xs, ys, xe, ye = box[:4]
            label_idx = int(box[5])
            confidence = box[4]
            if confidence > thresh[label_idx - 1]:
                score = int(confidence * 10 // 1)
                img = cv2.rectangle(img, (int(xs), int(ys)), (int(xe), int(ye)), color[label_idx], thickness)
                img = cv2.putText(img, str(score), (int(xs), int(ys - 20)),
                                  font, 3, color[label_idx], 5, cv2.LINE_AA)
                # img = cv2.putText(img, cfg['HL_map_anno'][label_idx] + '_' + str(score), (int(xs), int(ys - 20)),
                #                   font, 2, color[label_idx], 5, cv2.LINE_AA)
        save_name = name.replace('/', '_')
        cv2.imwrite(save_folder + '/' + save_name, img)


if __name__ == '__main__':
    set_name = 'test_set'  #choose the set to visualize

    vis_path = cfg['main_dir'] + '/results/visualization'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    save_folder = os.path.join(vis_path, set_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    gt_path = cfg[set_name]
    pre_path = cfg['main_dir'] + '/results/predictions/' + set_name + '.json'
    draw_box(gt_path, pre_path, [0.7, 0.7], save_folder)
