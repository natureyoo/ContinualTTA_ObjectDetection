# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Additionally modified by Yunqiu Xu for H2FA R-CNN
# ------------------------------------------------------------------------

import numpy as np
import os
from fvcore.common.file_io import PathManager
from PIL import Image
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode


CLASS_NAMES_20 = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

CLASS_NAMES_6 = ["bicycle", "bird", "car", "cat", "dog", "person"]

CLASS_NAMES_8 =["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


def load_cdod_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        if not os.path.isfile(anno_file):
            with Image.open(jpeg_file) as img:
                width, height = img.size
            r = {"file_name": jpeg_file, "image_id": fileid, "height": height, "width": width}
            instances = []
            r["annotations"] = instances
            dicts.append(r)
            continue

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # filter out some classes not in target domain
            if cls not in class_names:
                continue
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            difficult = int(obj.find("difficult").text)
            if difficult == 1:
                continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts

def register_cdod_pascal_voc(name, dirname, split, year, class_names):
    DatasetCatalog.register(name, lambda: load_cdod_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

def register_cdod_dataset(name, dirname, split, class_names):
    DatasetCatalog.register(name, lambda: load_cdod_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(thing_classes=list(class_names), dirname=dirname, split=split)


# ==== Predefined splits for cross domain object detection datasets =====
def register_all_cdod_datasets(root):
    SPLITS = [
        ("clipart1k_traintest", "clipart", "traintest"),
        ("clipart1k_train", "clipart", "train"),
        ("clipart1k_test", "clipart", "test"),
        ("watercolor2k_train", "watercolor", "train"),
        ("watercolor2k_test", "watercolor", "test"),
        ("watercolor2k_traintest", "watercolor", "traintest"),
        ("watercolor2k_extra", "watercolor", "extra"),
        ("watercolor2k_all", "watercolor", "all"),
        # ("comic2k_train", "comic", "train"),
        # ("comic2k_test", "comic", "test"),
        # ("comic2k_extra", "comic", "extra"),
        # ("cityscapes_source", "foggycityscapes", "train_s"),
        # ("foggycityscapes_target", "foggycityscapes", "train_t"),
        # ("foggycityscapes_test", "foggycityscapes", "test_t"),
    ]
    for name, dirname, split in SPLITS:
        if '1k' in name:
            class_name = CLASS_NAMES_20
        elif '2k' in name:
            class_name = CLASS_NAMES_6
        else:
            class_name = CLASS_NAMES_8
        register_cdod_dataset(name, os.path.join(root, dirname), split, class_name)
        MetadataCatalog.get(name).evaluator_type = "cross_domain"

    # VOC_SPLITS = [
    #     ("cdod_voc_2007_trainval", "VOC2007", "trainval"),
    #     ("cdod_voc_2007_test", "VOC2007", "test"),
    #     ("cdod_voc_2012_trainval", "VOC2012", "trainval"),
    # ]
    # for name, dirname, split in VOC_SPLITS:
    #     year = 2007 if "2007" in name else 2012
    #     register_cdod_pascal_voc(name, os.path.join(root, dirname), split, year, class_names=CLASS_NAMES_6)
    #     MetadataCatalog.get(name).evaluator_type = "cross_domain"

# Register them all under "./datasets
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_cdod_datasets(_root)
