# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

__all__ = ["load_kitti_instances", "register_kitti"]


# fmt: off
CLASS_NAMES = ( 'car', 'van', 'truck', 'person', 'person_sitting', 'cyclist', 'tram', 'misc')
# fmt: on


def load_kitti_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]], corrupt=None):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        #fileids = np.loadtxt(f, dtype=np.str)
        fileids = list(f.readlines())

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for _fileid in fileids:
        fileid = _fileid.split('.')[0]
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        if corrupt is not None:
            jpeg_file = os.path.join(dirname, "JPEGImages-{}".format(corrupt), fileid + ".png")
        else:
            jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".png")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        if corrupt in ['fog', 'rain']:
            x_start = r['width'] // 2 - 1216 // 2
            y_start = r['height'] // 2 - 352 // 2
            r['width'] = 1216
            r['height'] = 352
        instances = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text.lower()
            if cls == 'dontcare':
                continue
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            if corrupt in ['fog', 'rain']:
                bbox[0] -= x_start
                bbox[1] -= y_start
                bbox[2] -= x_start
                bbox[3] -= y_start
            # if corrupt in ['rain']:
            #     ori_height = int(tree.findall("./size/height")[0].text)
            #     ori_width = int(tree.findall("./size/width")[0].text)
            #     bbox[0] *= r['width'] / ori_width
            #     bbox[1] *= r['height'] / ori_height
            #     bbox[2] *= r['width'] / ori_width
            #     bbox[3] *= r['height'] / ori_height

            # bbox[0] -= 1.0
            # bbox[1] -= 1.0

            try: instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
            except:
                import pdb; pdb.set_trace()
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_kitti(name, dirname, split, class_names=CLASS_NAMES, corrupt=None):
    DatasetCatalog.register(name, lambda: load_kitti_instances(dirname, split, class_names, corrupt=corrupt))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, split=split, year=2012
    )
