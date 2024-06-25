import argparse
import json
import numpy as np
from detectron2.structures import Boxes, BoxMode, Instances


def create_pred_instances(prediction, output_size, image_size, conf_th):
    ret = Instances(image_size)

    score = np.asarray(prediction.scores.cpu())
    chosen = (score > conf_th).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([prediction.pred_boxes[int(i)].tensor.cpu().numpy() for i in chosen]).reshape(-1, 4)
    bbox[:, 0] = bbox[:, 0] * image_size[1] / output_size[1]
    bbox[:, 2] = bbox[:, 2] * image_size[1] / output_size[1]
    bbox[:, 1] = bbox[:, 1] * image_size[0] / output_size[0]
    bbox[:, 3] = bbox[:, 3] * image_size[0] / output_size[0]
    # bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([prediction.pred_classes[i].item() for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    return ret
